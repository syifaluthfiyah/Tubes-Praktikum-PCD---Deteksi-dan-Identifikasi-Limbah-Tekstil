import cv2
import numpy as np
import mahotas

# Ekstraksi fitur warna: rata-rata saturasi HSV
def extract_color_feature(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    mean_s = np.mean(s) / 255.0
    return mean_s

# Ekstraksi fitur bentuk: luas kontur terbesar (thresholding)
def extract_shape_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    max_area = max(cv2.contourArea(c) for c in contours)
    return max_area

# Ekstraksi fitur tekstur: rata-rata Haralick
def extract_texture_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick_features = mahotas.features.haralick(gray)
    mean_haralick = haralick_features.mean()
    return mean_haralick

# Rule-based klasifikasi
def classify_waste(color_feat, shape_feat, texture_feat):
    if shape_feat < 5000 and color_feat > 0.5 and texture_feat < 20:
        return "Tidak Tercemar"
    elif shape_feat < 15000 and color_feat <= 0.5 and texture_feat < 30:
        return "Tercemar Ringan"
    else:
        return "Tercemar Berat"

def main():
    print("Tekan 'o' untuk membuka gambar, 'q' untuk keluar.")

    img = None

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('o'):
            # Open file dialog untuk memilih gambar
            import tkinter.filedialog as fd
            file_path = fd.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
            if file_path:
                img = cv2.imread(file_path)
                if img is None:
                    print("Gagal membuka gambar!")
                    continue

                # Tampilkan gambar asli
                cv2.imshow("Gambar Limbah Cair", img)

                # Ekstrak fitur
                color_feat = extract_color_feature(img)
                shape_feat = extract_shape_feature(img)
                texture_feat = extract_texture_feature(img)

                kelas = classify_waste(color_feat, shape_feat, texture_feat)

                print(f"\nHasil Deteksi Limbah:")
                print(f" - Warna (Saturasi rata-rata): {color_feat:.2f}")
                print(f" - Bentuk (Area kontur): {shape_feat:.0f}")
                print(f" - Tekstur (Haralick mean): {texture_feat:.2f}")
                print(f" => Kategori: {kelas}\n")

                # Tampilkan teks di jendela gambar
                img_copy = img.copy()
                cv2.putText(img_copy, f"Kategori: {kelas}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Gambar Limbah Cair", img_copy)

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()