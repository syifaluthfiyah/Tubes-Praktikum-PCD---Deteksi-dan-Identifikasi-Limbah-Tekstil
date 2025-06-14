import cv2
import numpy as np
from skimage import feature, filters, measure
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

class FeatureExtractor:
    """Class untuk ekstraksi fitur warna dengan multi-color masking limbah cair"""
    
    def __init__(self):
        # Rentang HSV untuk berbagai warna limbah cair
        self.color_ranges_dict = {
            "Coklat/Oranye": [(np.array([10, 100, 20]), np.array([30, 255, 255]))],
            "Biru": [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            "Merah": [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ],
            "Semua Warna": [
                (np.array([10, 100, 20]), np.array([30, 255, 255])),
                (np.array([100, 50, 50]), np.array([130, 255, 255])),
                (np.array([40, 40, 40]), np.array([90, 255, 255])),
                (np.array([20, 100, 100]), np.array([40, 255, 255])),
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ]
        }
    
    def extract_color_features(self, image, mask_color="Semua Warna"):
        """Ekstraksi fitur warna dengan masking sesuai pilihan warna"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Pilih rentang warna sesuai pilihan user
        color_ranges = self.color_ranges_dict.get(mask_color, self.color_ranges_dict["Semua Warna"])
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        total_pixels = image.shape[0] * image.shape[1]
        waste_pixels = cv2.countNonZero(combined_mask)
        waste_percentage = (waste_pixels / total_pixels) * 100
        hist_h = cv2.calcHist([hsv], [0], combined_mask, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], combined_mask, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], combined_mask, [256], [0, 256])
        hist_h = cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
        hist_s = cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
        hist_v = cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)
        mean_hsv = cv2.mean(hsv, mask=combined_mask)
        std_hsv = cv2.meanStdDev(hsv, mask=combined_mask)[1]
        color_features = np.concatenate([
            hist_h.flatten(), hist_s.flatten(), hist_v.flatten(),
            mean_hsv, std_hsv.flatten(), [waste_percentage]
        ])
        return color_features, combined_mask
    
    def extract_all_features(self, image, mask_color="Semua Warna"):
        """Ekstraksi fitur dengan fokus pada warna coklat/oranye"""
        return self.extract_color_features(image, mask_color)

def detect_jeans_shape(contour, gray_img):
    # Ekstraksi fitur bentuk utama
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(h) / w if w != 0 else 0
    area = cv2.contourArea(contour)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area != 0 else 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0
    # Rule baru: semua gambar di folder Jeans dianggap jeans
    jeans_label = "Jeans Bagus"
    # --- Canny Edge + Contour Filtering untuk deteksi robekan ---
    roi = gray_img[y:y+h, x:x+w]
    # 1. Canny edge detection pada ROI jeans
    edges = cv2.Canny(roi, 80, 180)
    # 2. Morphological closing untuk menutup edge robekan yang putus-putus
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # 3. Cari kontur pada hasil morphology
    holes, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rusak = False
    rusak_boxes = []
    for hole in holes:
        hole_area = cv2.contourArea(hole)
        if hole_area > 100:  # Menurunkan threshold area robekan agar lebih sensitif lagi
            rusak = True
            hx, hy, hw, hh = cv2.boundingRect(hole)
            rusak_boxes.append((x+hx, y+hy, hw, hh))
    if rusak:
        jeans_label = "Jeans Rusak"
    return jeans_label, aspect_ratio, extent, solidity, rusak_boxes

def detect_texture_type(gray_img, mask=None):
    # Jika mask diberikan, hanya analisis area mask
    if mask is not None:
        gray = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    else:
        gray = gray_img

    # Pastikan ROI tidak kosong
    if gray.shape[0] == 0 or gray.shape[1] == 0 or np.all(gray == 0):
        return "Tidak terklasifikasi", 0, 0, 0, np.zeros(11) # Return dummy values

    # GLCM
    # Tingkatkan jumlah angle untuk analisis tekstur yang lebih robust
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    contrast = np.mean([graycoprops(glcm, 'contrast')[0, i] for i in range(4)])
    homogeneity = np.mean([graycoprops(glcm, 'homogeneity')[0, i] for i in range(4)])
    energy = np.mean([graycoprops(glcm, 'energy')[0, i] for i in range(4)])
    
    # LBP
    # Gunakan parameter yang lebih sesuai untuk LBP, radius dan points bisa disetel di SettingsDialog
    # Saat ini, saya akan menggunakan nilai default yang lebih robust
    lbp_radius = 3  # Radius lebih kecil, fokus pada detail lokal
    lbp_points = lbp_radius * 8 # Jumlah titik tetangga

    # Pastikan LBP tidak error untuk gambar yang terlalu kecil
    if gray.shape[0] < lbp_radius * 2 + 1 or gray.shape[1] < lbp_radius * 2 + 1:
        return "Tidak terklasifikasi", contrast, homogeneity, energy, np.zeros(lbp_points + 3) # Ukuran hist disesuaikan

    lbp = local_binary_pattern(gray, lbp_points, lbp_radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(lbp_points + 3), range=(0, lbp_points + 2))
    hist = hist.astype("float") / (hist.sum() + 1e-6)
    
    # --- Log detail fitur untuk debugging ---
    print(f"[TEKSTUR DEBUG] Contrast: {contrast:.2f}, Homogeneity: {homogeneity:.2f}, Energy: {energy:.2f}")
    print(f"[TEKSTUR DEBUG] LBP Hist (bins 0-9, or max_points+2): {hist[:10]}") # Tampilkan lebih banyak bins
    
    # Rule-based logic plastik vs jeans (DIOPTIMALKAN LAGI):
    # Plastik: contrast sangat rendah, homogeneity sangat tinggi, energy tinggi, LBP bin 0/1 tinggi
    # Jeans: contrast tinggi, homogeneity rendah, LBP bin > 1 tinggi (non-uniform patterns)
    
    is_plastic = False
    is_jeans = False

    # Rule untuk Plastik: Lebih ketat pada homogeneity dan contrast yang sangat rendah
    # Memperhatikan bahwa LBP bin 0/1 harus dominan
    if (homogeneity > 0.88 and contrast < 100 and energy > 0.25) and \
       (hist[0] > 0.5 or hist[1] > 0.5): # LBP peaks for uniform/near-uniform patterns sangat kuat
        is_plastic = True
            
    # Rule untuk Jeans: Lebih fokus pada contrast yang lebih tinggi dan homogeneity yang lebih rendah
    # Memperhatikan pola LBP non-uniform
    if (contrast > 200 and homogeneity < 0.6) and \
       (np.sum(hist[2:9]) > 0.15): # Total LBP bins 2-8 cukup signifikan, menunjukkan tekstur non-uniform
        is_jeans = True

    # Resolusi konflik/prioritas:
    if is_plastic and is_jeans: # Jika keduanya terdeteksi, mungkin ada tumpang tindih atau ambiguitas
        # Jika kedua kondisi sangat kuat, ini mungkin objek yang ambigu atau perlu aturan lebih lanjut.
        # Untuk saat ini, jika terjadi tumpang tindih dan tidak ada keunggulan jelas, anggap tidak terklasifikasi.
        if (homogeneity > 0.95 and contrast < 50): # Sangat plastik
            label = "Plastik"
        elif (contrast > 3000 and np.sum(hist[2:9]) > 0.3): # Sangat jeans
            label = "Jeans"
        else:
            label = "Tidak terklasifikasi" # Masih ambigu atau berada di perbatasan
    elif is_plastic and not is_jeans:
        label = "Plastik"
    elif is_jeans and not is_plastic:
        label = "Jeans"
    else:
        label = "Tidak terklasifikasi"
            
    return label, contrast, homogeneity, energy, hist

def process_image_for_display(image_bgr, mode="cair", feature_extractor=None, mask_color_combo_currentText=None):
    """Fungsi terintegrasi untuk pemrosesan citra dan deteksi"""
    # Konversi ke grayscale untuk mode bentuk dan tekstur
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    result_image = image_bgr.copy()
    
    # List untuk menyimpan semua gambar yang akan ditampilkan di processed image tab
    images_for_plot = []
    images_for_plot.append((cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), "Original Image"))
    
    if mode == "cair":
        # Mode limbah cair - deteksi berdasarkan warna
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask_color = mask_color_combo_currentText if mask_color_combo_currentText is not None else "Semua Warna"
        color_ranges = feature_extractor.color_ranges_dict.get(mask_color, 
                                                                  feature_extractor.color_ranges_dict["Semua Warna"])
        
        # Buat mask kombinasi
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        images_for_plot.append((cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB), "HSV Mask"))
        
        # Deteksi kontur
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Visualisasi hasil
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 255), 4) # Ketebalan 4
                cv2.putText(result_image, f"Limbah Cair: {area:.0f}px", (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        images_for_plot.append((cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), "Detection Result"))
        
        return cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), images_for_plot
        
    elif mode == "bentuk":
        # Mode bentuk - deteksi jeans rusak vs bagus
        images_for_plot.append((cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), "Grayscale"))
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        images_for_plot.append((cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB), "Binary/Thresholding"))
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100: # Threshold area minimum untuk objek yang terdeteksi
                x, y, w, h = cv2.boundingRect(contour)
                # Panggil detect_jeans_shape untuk klasifikasi dan deteksi robekan
                label, aspect_ratio, extent, solidity, rusak_boxes = detect_jeans_shape(contour, gray)
                
                color = (0, 255, 0) if label == "Jeans Bagus" else (0, 0, 255) # Hijau untuk bagus, Merah untuk rusak
                
                # Bounding box utama jeans
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 4) # Ketebalan 4
                # Adjust text position slightly to avoid being cut off if box is at edge
                text_y = y - 10 if y - 10 > 10 else y + h + 20 # Move below if too high
                cv2.putText(result_image, label, (x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Bounding box untuk area rusak (robekan)
                for (hx, hy, hw, hh) in rusak_boxes:
                    # Garis putus-putus kuning untuk robekan
                    for i in range(0, hw, 5):
                        cv2.line(result_image, 
                                (x+hx+i, y+hy), 
                                (x+hx+min(i+3, hw), y+hy), 
                                (0, 255, 255), 2)
                        cv2.line(result_image, 
                                (x+hx+i, y+hy+hh), 
                                (x+hx+min(i+3, hw), y+hy+hh), 
                                (0, 255, 255), 2)
                    for i in range(0, hh, 5):
                        cv2.line(result_image, 
                                (x+hx, y+hy+i), 
                                (x+hx, y+hy+min(i+3, hh)), 
                                (0, 255, 255), 2)
                        cv2.line(result_image, 
                                (x+hx+hw, y+hy+i), 
                                (x+hx+hw, y+hy+min(i+3, hh)), 
                                (0, 255, 255), 2)
                    text_y_rob = y + hy - 10 if y + hy - 10 > 10 else y + hy + hh + 20
                    cv2.putText(result_image, "Robekan", (x+hx, text_y_rob),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        images_for_plot.append((cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), "Detection Result"))
        
        return cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), images_for_plot
        
    else:  # mode == "tekstur"
        # Mode tekstur - deteksi plastik vs jeans
        images_for_plot.append((cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), "Grayscale"))
        
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # LBP untuk visualisasi
        lbp = feature.local_binary_pattern(gray, 16, 2, method='uniform')
        lbp_img = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min()))
        images_for_plot.append((cv2.cvtColor(lbp_img, cv2.COLOR_GRAY2RGB), "LBP Image"))
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100: # Threshold area minimum untuk objek yang terdeteksi
                x, y, w, h = cv2.boundingRect(contour)
                roi = gray[y:y+h, x:x+w]
                
                # Panggil detect_texture_type untuk klasifikasi plastik vs jeans
                label, contrast, homogeneity, energy, hist = detect_texture_type(roi) # Tidak perlu mask di sini lagi
                
                color = (0, 255, 0) if label == "Plastik" else ((0, 0, 255) if label == "Jeans" else (255, 255, 0)) # Hijau utk Plastik, Merah utk Jeans, Kuning utk Tidak terklasifikasi
                
                # Bounding box dan label
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 4) # Ketebalan 4
                # Adjust text position slightly to avoid being cut off if box is at edge
                text_y = y - 10 if y - 10 > 10 else y + h + 20 # Move below if too high
                cv2.putText(result_image, label, (x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        images_for_plot.append((cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), "Detection Result"))
        
        return cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), images_for_plot 