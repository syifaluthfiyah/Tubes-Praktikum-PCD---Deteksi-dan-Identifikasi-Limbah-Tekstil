import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from skimage import feature, filters, measure
from skimage.color import rgb2gray, rgb2hsv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
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
    
    def extract_shape_features(self, image):
        """Fitur bentuk dinonaktifkan"""
        return np.zeros(10)
    
    def extract_texture_features(self, image):
        """Fitur tekstur dinonaktifkan"""
        return np.zeros(26)
    
    def extract_all_features(self, image, mask_color="Semua Warna"):
        """Ekstraksi fitur dengan fokus pada warna coklat/oranye"""
        return self.extract_color_features(image, mask_color)

class TextileWasteDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.current_image = None
        self.processed_image = None
        self.selected_feature_type = "Warna (Limbah Cair)"  # Default
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Liquid Textile Waste Detection System - Manual Edition')
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b3e50;
                color: white;
            }
            QWidget {
                background-color: #34495e;
                color: white;
                border: 1px solid #3498db;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #3498db;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f5f8b;
            }
            QLabel {
                border: none;
                padding: 5px;
            }
            QTextEdit {
                background-color: #2c3e50;
                border: 1px solid #3498db;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            QTableWidget {
                gridline-color: #3498db;
                background-color: #2c3e50;
                alternate-background-color: #34495e;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                padding: 5px;
                border: none;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 3)
        
        # Status bar
        self.statusBar().showMessage('System Ready - Select an image to begin liquid textile waste detection')
        self.statusBar().setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        
    def create_left_panel(self):
        """Create left control panel"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Image Selection Group
        image_group = QGroupBox("📁 Image Selection")
        image_layout = QVBoxLayout(image_group)
        
        self.image_status = QLabel("No image selected")
        self.image_status.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_status)
        
        self.browse_btn = QPushButton("📂 Browse Image")
        self.browse_btn.clicked.connect(self.browse_image)
        image_layout.addWidget(self.browse_btn)
        
        left_layout.addWidget(image_group)
        
        # Masking Color Selection
        color_group = QGroupBox("🎨 Masking Color Selection")
        color_layout = QVBoxLayout(color_group)
        self.mask_color_combo = QComboBox()
        self.mask_color_combo.addItems(["Coklat/Oranye", "Biru", "Merah", "Semua Warna"])
        self.mask_color_combo.setCurrentIndex(3)
        color_layout.addWidget(QLabel("Pilih warna limbah cair yang ingin dideteksi:"))
        color_layout.addWidget(self.mask_color_combo)
        left_layout.addWidget(color_group)
        color_group.hide()  # Default: hide, akan di-show jika perlu
        
        # Processing Controls Group
        processing_group = QGroupBox("⚙️ Processing Controls")
        processing_layout = QVBoxLayout(processing_group)
        
        self.detect_btn = QPushButton("🔍 Start Liquid Detection")
        self.detect_btn.clicked.connect(self.start_detection)
        self.detect_btn.setEnabled(False)
        processing_layout.addWidget(self.detect_btn)
        
        self.settings_btn = QPushButton("⚙️ Open Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        processing_layout.addWidget(self.settings_btn)
        
        left_layout.addWidget(processing_group)
        
        # Processing Log Group
        log_group = QGroupBox("📋 Processing Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        self.clear_log_btn = QPushButton("🗑️ Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(self.clear_log_btn)
        
        left_layout.addWidget(log_group)
        
        left_layout.addStretch() # Removed Training Controls Group
        return left_widget
    
    def create_right_panel(self):
        """Create right display panel"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3498db;
                background-color: #34495e;
            }
            QTabBar::tab {
                background-color: #2c3e50;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
            }
        """)
        
        # Detection Result Tab
        self.detection_tab = self.create_detection_tab()
        self.tab_widget.addTab(self.detection_tab, "🔍 Liquid Detection Result")
        
        # Processed Image Tab
        self.processed_tab = self.create_processed_tab()
        self.tab_widget.addTab(self.processed_tab, "🖼️ Processed Image")
        
        right_layout.addWidget(self.tab_widget)
        
        return right_widget
    
    def create_detection_tab(self):
        """Create detection results tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Top section with summary and statistics
        top_section = QHBoxLayout()
        
        # Detection Summary
        summary_group = QGroupBox("🎯 Liquid Detection Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_label = QLabel("No detection performed yet")
        self.summary_label.setAlignment(Qt.AlignCenter)
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        top_section.addWidget(summary_group)
        
        # Statistics
        stats_group = QGroupBox("📊 Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel("Waiting for analysis...")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        top_section.addWidget(stats_group)
        
        layout.addLayout(top_section)
        
        # Image comparison section
        image_section = QHBoxLayout()
        
        # Comparison Image
        comp_group = QGroupBox("🖼️ Original Image")
        comp_layout = QVBoxLayout(comp_group)
        self.comparison_label = QLabel("No image loaded")
        self.comparison_label.setAlignment(Qt.AlignCenter)
        self.comparison_label.setMinimumSize(400, 300)
        self.comparison_label.setStyleSheet("border: 2px dashed #3498db;")
        comp_layout.addWidget(self.comparison_label)
        image_section.addWidget(comp_group)
        
        # Detection Result Image
        result_group = QGroupBox("❌ Liquid Detection Result")
        result_layout = QVBoxLayout(result_group)
        self.result_label = QLabel("No image loaded")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(400, 300)
        self.result_label.setStyleSheet("border: 2px dashed #e74c3c;")
        result_layout.addWidget(self.result_label)
        image_section.addWidget(result_group)
        
        layout.addLayout(image_section)
        
        # Detailed Analysis Table
        analysis_group = QGroupBox("📋 Detailed Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.analysis_table = QTableWidget()
        self.analysis_table.setColumnCount(6)
        self.analysis_table.setHorizontalHeaderLabels([
            "Object ID", "Confidence", "Classification", "Position", "Size", "Area"
        ])
        self.analysis_table.horizontalHeader().setStretchLastSection(True)
        analysis_layout.addWidget(self.analysis_table)
        
        layout.addWidget(analysis_group)
        
        return tab
    
    def create_processed_tab(self):
        """Create processed image tab dengan multi-preview"""
        tab = QWidget()
        self.processed_layout = QVBoxLayout(tab)
        # Layout untuk gambar proses
        self.processed_images_layout = QHBoxLayout()
        self.processed_layout.addLayout(self.processed_images_layout)
        # Simpan referensi label gambar dan label teks
        self.processed_image_labels = []
        self.processed_image_titles = []
        return tab
    
    def browse_image(self):
        """Browse and load image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image for Liquid Textile Waste Detection", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            try:
                # Load image as BGR (as is from cv2)
                self.current_image_bgr = cv2.imread(file_path)
                # Simpan juga versi RGB untuk tampilan
                self.current_image = cv2.cvtColor(self.current_image_bgr, cv2.COLOR_BGR2RGB)
                # Update UI
                self.image_status.setText(f"Image loaded: {os.path.basename(file_path)}")
                self.detect_btn.setEnabled(True)
                # Display image (RGB) di comparison_label
                self.display_image(self.current_image, self.comparison_label)
                self.log_message(f"Image loaded successfully: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
                self.log_message(f"Error loading image: {str(e)}")
    
    def display_image(self, image, label):
        """Display image in label"""
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        
        # Resize image to fit label
        label_size = label.size()
        if width > label_size.width() or height > label_size.height():
            # Calculate scale factor
            scale_w = label_size.width() / width
            scale_h = label_size.height() / height
            scale = min(scale_w, scale_h)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            image = cv2.resize(image, (new_width, new_height))
            height, width, channel = image.shape
            bytes_per_line = 3 * width
        
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
    
    def detect_jeans_shape(self, contour, gray_img):
        # Ekstraksi fitur bentuk utama
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(h) / w if w != 0 else 0
        area = cv2.contourArea(contour)
        rect_area = w * h
        extent = float(area) / rect_area if rect_area != 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        # Deteksi dua cabang bawah (kaki jeans) secara sederhana
        # Ambil bagian bawah bounding box
        jeans_label = "Bukan Jeans"
        if aspect_ratio > 1.2 and extent > 0.5 and 0.85 < solidity < 1.0:
            # Cek simetri bawah (opsional, sederhana)
            bottom = y + h - int(0.2 * h)
            jeans_roi = gray_img[bottom:y+h, x:x+w]
            mid = jeans_roi.shape[1] // 2
            left = jeans_roi[:, :mid]
            right = jeans_roi[:, mid:]
            left_sum = np.sum(left > 0)
            right_sum = np.sum(right > 0)
            symmetry = min(left_sum, right_sum) / max(left_sum, right_sum) if max(left_sum, right_sum) > 0 else 0
            # Deteksi rusak jika simetri buruk atau area bawah bolong
            if symmetry > 0.7:
                jeans_label = "Jeans Bagus"
            else:
                jeans_label = "Jeans Rusak"
        return jeans_label, aspect_ratio, extent, solidity

    def detect_texture_type(self, gray_img, mask=None):
        # Jika mask diberikan, hanya analisis area mask
        if mask is not None:
            gray = cv2.bitwise_and(gray_img, gray_img, mask=mask)
        else:
            gray = gray_img
        # GLCM
        glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        # LBP
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(11), range=(0, 10))
        hist = hist.astype("float") / (hist.sum() + 1e-6)
        # Rule-based logic
        if contrast > 2.0 and homogeneity < 0.5 and hist[3] > 0.15:
            label = "Tekstil (Jeans)"
        elif contrast < 1.0 and homogeneity > 0.7:
            label = "Plastik"
        else:
            label = "Tidak terklasifikasi"
        return label, contrast, homogeneity, energy, hist

    def process_image_for_display(self, image_bgr, mode="cair"):
        if mode == "cair":
            mask_color = self.mask_color_combo.currentText() if hasattr(self, 'mask_color_combo') else "Semua Warna"
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            color_ranges = self.feature_extractor.color_ranges_dict.get(mask_color, self.feature_extractor.color_ranges_dict["Semua Warna"])
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in color_ranges:
                mask = cv2.inRange(hsv, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result_image = image_bgr.copy()
            cv2.drawContours(result_image, contours, -1, (0, 140, 255), 2)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(result_image, f"Waste Area: {area:.0f}px", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            return result_image_rgb, combined_mask, None
        elif mode == "bentuk":
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result_image = image_bgr.copy()
            jeans_result = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    label, aspect_ratio, extent, solidity = self.detect_jeans_shape(contour, gray)
                    color = (0, 255, 0) if label == "Jeans Bagus" else ((0, 0, 255) if label == "Jeans Rusak" else (255, 255, 0))
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    jeans_result.append({
                        'label': label,
                        'aspect_ratio': aspect_ratio,
                        'extent': extent,
                        'solidity': solidity,
                        'area': area
                    })
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            # Simpan hasil deteksi jeans untuk analisis tabel
            self.last_jeans_result = jeans_result
            return result_image_rgb, binary, gray
        else:  # mode == "tekstur"
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            # LBP
            lbp = feature.local_binary_pattern(gray, 24, 8, method='uniform')
            lbp_img = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min()))
            lbp_img_rgb = cv2.cvtColor(lbp_img, cv2.COLOR_GRAY2RGB)
            # Thresholding untuk bounding box
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result_image = image_bgr.copy()
            texture_result = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = gray[y:y+h, x:x+w]
                    roi_mask = binary[y:y+h, x:x+w]
                    label, contrast, homogeneity, energy, hist = self.detect_texture_type(roi, mask=roi_mask)
                    color = (0, 255, 0) if label == "Tekstil (Jeans)" else ((0, 0, 255) if label == "Plastik" else (255, 255, 0))
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    texture_result.append({
                        'label': label,
                        'contrast': contrast,
                        'homogeneity': homogeneity,
                        'energy': energy,
                        'hist3': hist[3],
                        'area': area
                    })
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            self.last_texture_result = texture_result
            return result_image_rgb, binary, lbp_img_rgb
    
    def start_detection(self):
        if not hasattr(self, 'current_image_bgr') or self.current_image_bgr is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
        try:
            self.log_message("Starting detection process (manual feature extraction)...")
            mode = "cair" if self.selected_feature_type == "Warna (Limbah Cair)" else (
                "bentuk" if self.selected_feature_type == "Bentuk (Limbah Padat)" else "tekstur")
            self.processed_image, self.last_mask, self.extra_img = self.process_image_for_display(self.current_image_bgr.copy(), mode=mode)
            self.update_detection_results_manual()
            self.log_message("Detection completed successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")
            self.log_message(f"Detection error: {str(e)}")
    
    def update_detection_results_manual(self):
        """Update detection results in UI untuk multi-preview processed image"""
        # Update summary
        summary_text = """Detection Results:\n"""
        if self.processed_image is not None:
            summary_text += """Visual detection performed focusing on liquid textile waste areas.\n"""
        else:
            summary_text += """No processed image available for visual detection."""
        self.summary_label.setText(summary_text)
        # Update statistics
        stats_text = """Analysis based on HSV color model:\n"""
        stats_text += """Features Extracted: Multi-Color Liquid Waste Detection"""
        self.stats_label.setText(stats_text)
        # Display processed image (bounding box) di result_label
        if self.processed_image is not None:
            self.display_image(self.processed_image, self.result_label)
        # Tampilkan multi-preview di processed_image_label
        self.show_processed_images()
        # Update analysis table
        self.update_analysis_table_manual()
    
    def show_processed_images(self):
        # Bersihkan layout
        for i in reversed(range(self.processed_images_layout.count())):
            widget = self.processed_images_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        self.processed_image_labels = []
        self.processed_image_titles = []
        mode = "cair" if self.selected_feature_type == "Warna (Limbah Cair)" else (
            "bentuk" if self.selected_feature_type == "Bentuk (Limbah Padat)" else "tekstur")
        images = []
        titles = []
        if mode == "cair":
            orig = self.current_image if self.current_image is not None else None
            mask = self.last_mask if hasattr(self, 'last_mask') else None
            det = self.processed_image if self.processed_image is not None else None
            if orig is not None:
                images.append(orig)
                titles.append("Original Image")
            if mask is not None:
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                images.append(mask_rgb)
                titles.append("HSV Mask")
            if det is not None:
                images.append(det)
                titles.append("Detection Result")
        elif mode == "bentuk":
            orig = self.current_image if self.current_image is not None else None
            gray = self.extra_img if hasattr(self, 'extra_img') and self.extra_img is not None else None
            binary = self.last_mask if hasattr(self, 'last_mask') else None
            det = self.processed_image if self.processed_image is not None else None
            if orig is not None:
                images.append(orig)
                titles.append("Original Image")
            if gray is not None:
                gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                images.append(gray_rgb)
                titles.append("Grayscale")
            if binary is not None:
                binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                images.append(binary_rgb)
                titles.append("Binary/Thresholding")
            if det is not None:
                images.append(det)
                titles.append("Detection Result")
        else:  # tekstur
            orig = self.current_image if self.current_image is not None else None
            gray = cv2.cvtColor(self.current_image_bgr, cv2.COLOR_BGR2GRAY) if hasattr(self, 'current_image_bgr') else None
            lbp_img = self.extra_img if hasattr(self, 'extra_img') and self.extra_img is not None else None
            det = self.processed_image if self.processed_image is not None else None
            if orig is not None:
                images.append(orig)
                titles.append("Original Image")
            if gray is not None:
                gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                images.append(gray_rgb)
                titles.append("Grayscale")
            if lbp_img is not None:
                images.append(lbp_img)
                titles.append("LBP Image")
            if det is not None:
                images.append(det)
                titles.append("Detection Result")
        for img, title in zip(images, titles):
            vbox = QVBoxLayout()
            label_img = QLabel()
            label_img.setAlignment(Qt.AlignCenter)
            h, w = img.shape[:2]
            max_w, max_h = 300, 220
            scale = min(max_w / w, max_h / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            img_disp = cv2.resize(img, (new_w, new_h))
            qimg = QImage(img_disp.data, new_w, new_h, 3 * new_w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            label_img.setPixmap(pixmap)
            label_title = QLabel(title)
            label_title.setAlignment(Qt.AlignCenter)
            label_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #3498db;")
            vbox.addWidget(label_img)
            vbox.addWidget(label_title)
            container = QWidget()
            container.setLayout(vbox)
            self.processed_images_layout.addWidget(container)
    
    def update_analysis_table_manual(self):
        if self.current_image is not None:
            if self.selected_feature_type == "Warna (Limbah Cair)":
                features, mask = self.feature_extractor.extract_all_features(self.current_image_bgr, mask_color=self.mask_color_combo.currentText())
                total_pixels = self.current_image_bgr.shape[0] * self.current_image_bgr.shape[1]
                waste_pixels = cv2.countNonZero(mask)
                waste_percentage = (waste_pixels / total_pixels) * 100
                self.analysis_table.setRowCount(1)
                self.analysis_table.setItem(0, 0, QTableWidgetItem("Image_01"))
                self.analysis_table.setItem(0, 1, QTableWidgetItem(f"{waste_percentage:.1f}%"))
                self.analysis_table.setItem(0, 2, QTableWidgetItem("Liquid Textile Waste"))
                self.analysis_table.setItem(0, 3, QTableWidgetItem("N/A"))
                self.analysis_table.setItem(0, 4, QTableWidgetItem(f"Approx. {self.current_image_bgr.shape[1]}x{self.current_image_bgr.shape[0]}"))
                self.analysis_table.setItem(0, 5, QTableWidgetItem(f"{waste_pixels} pixels"))
            elif self.selected_feature_type == "Bentuk (Limbah Padat)":
                # Tampilkan hasil deteksi jeans
                jeans_result = getattr(self, 'last_jeans_result', [])
                self.analysis_table.setRowCount(max(1, len(jeans_result)))
                if jeans_result:
                    for i, res in enumerate(jeans_result):
                        self.analysis_table.setItem(i, 0, QTableWidgetItem(f"Objek_{i+1}"))
                        self.analysis_table.setItem(i, 1, QTableWidgetItem("-"))
                        self.analysis_table.setItem(i, 2, QTableWidgetItem(res['label']))
                        self.analysis_table.setItem(i, 3, QTableWidgetItem(f"AR: {res['aspect_ratio']:.2f}"))
                        self.analysis_table.setItem(i, 4, QTableWidgetItem(f"Extent: {res['extent']:.2f}"))
                        self.analysis_table.setItem(i, 5, QTableWidgetItem(f"Solidity: {res['solidity']:.2f}"))
                else:
                    self.analysis_table.setItem(0, 0, QTableWidgetItem("-"))
                    self.analysis_table.setItem(0, 1, QTableWidgetItem("-"))
                    self.analysis_table.setItem(0, 2, QTableWidgetItem("Tidak ada objek"))
                    self.analysis_table.setItem(0, 3, QTableWidgetItem("-"))
                    self.analysis_table.setItem(0, 4, QTableWidgetItem("-"))
                    self.analysis_table.setItem(0, 5, QTableWidgetItem("-"))
            else:  # Tekstur
                texture_result = getattr(self, 'last_texture_result', [])
                self.analysis_table.setRowCount(max(1, len(texture_result)))
                if texture_result:
                    for i, res in enumerate(texture_result):
                        self.analysis_table.setItem(i, 0, QTableWidgetItem(f"Objek_{i+1}"))
                        self.analysis_table.setItem(i, 1, QTableWidgetItem("-"))
                        self.analysis_table.setItem(i, 2, QTableWidgetItem(res['label']))
                        self.analysis_table.setItem(i, 3, QTableWidgetItem(f"Contrast: {res['contrast']:.2f}"))
                        self.analysis_table.setItem(i, 4, QTableWidgetItem(f"Homog: {res['homogeneity']:.2f}"))
                        self.analysis_table.setItem(i, 5, QTableWidgetItem(f"LBP3: {res['hist3']:.2f}"))
                else:
                    self.analysis_table.setItem(0, 0, QTableWidgetItem("-"))
                    self.analysis_table.setItem(0, 1, QTableWidgetItem("-"))
                    self.analysis_table.setItem(0, 2, QTableWidgetItem("Tidak ada objek"))
                    self.analysis_table.setItem(0, 3, QTableWidgetItem("-"))
                    self.analysis_table.setItem(0, 4, QTableWidgetItem("-"))
                    self.analysis_table.setItem(0, 5, QTableWidgetItem("-"))
        else:
            self.analysis_table.setRowCount(0)
    
    # Removed load_dataset, train_models, save_models, load_models
    
    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self)
        # Set combo sesuai state terakhir
        dialog.feature_type_combo.setCurrentText(self.selected_feature_type)
        dialog.exec_()
        # Simpan pilihan setelah dialog ditutup
        self.selected_feature_type = dialog.feature_type_combo.currentText()
        self.update_masking_color_visibility()
    
    def update_masking_color_visibility(self):
        # Tampilkan/hilangkan Masking Color Selection di panel kiri
        if hasattr(self, 'mask_color_combo'):
            if self.selected_feature_type == "Warna (Limbah Cair)":
                self.mask_color_combo.parentWidget().show()
            else:
                self.mask_color_combo.parentWidget().hide()
    
    def clear_log(self):
        """Clear processing log"""
        self.log_text.clear()
        self.log_message("Log cleared")
    
    def log_message(self, message):
        """Add message to log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Settings")
        self.setModal(True)
        self.setFixedSize(600, 500)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #2b3e50;
                color: white;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            QPushButton {
                background-color: #3498db;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #34495e;
                border: 1px solid #3498db;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Feature Extraction Settings
        feature_group = QGroupBox("Feature Extraction Settings")
        feature_layout = QFormLayout(feature_group)
        
        # ComboBox untuk memilih ekstraksi fitur
        self.feature_type_combo = QComboBox()
        self.feature_type_combo.addItems([
            "Warna (Limbah Cair)",
            "Bentuk (Limbah Padat)",
            "Tekstur (Limbah Padat)"
        ])
        self.feature_type_combo.setCurrentIndex(0)
        feature_layout.addRow("Ekstraksi Fitur:", self.feature_type_combo)
        
        self.color_bins = QSpinBox()
        self.color_bins.setRange(8, 64)
        self.color_bins.setValue(32)
        feature_layout.addRow("Color Histogram Bins:", self.color_bins)
        
        self.lbp_radius = QSpinBox()
        self.lbp_radius.setRange(1, 10)
        self.lbp_radius.setValue(8)
        feature_layout.addRow("LBP Radius:", self.lbp_radius)
        
        self.lbp_points = QSpinBox()
        self.lbp_points.setRange(8, 32)
        self.lbp_points.setValue(24)
        feature_layout.addRow("LBP Points:", self.lbp_points)
        
        layout.addWidget(feature_group)
        
        # Classification Settings (Simplified, only confidence threshold remains relevant for visual feedback)
        class_group = QGroupBox("Visual Interpretation Settings")
        class_layout = QFormLayout(class_group)
        
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(0.1, 1.0)
        self.confidence_threshold.setSingleStep(0.1)
        self.confidence_threshold.setValue(0.7) # Can still be used for internal filtering if desired
        class_layout.addRow("Minimum Object Area (%):", self.confidence_threshold) # Re-purposing for visual feedback
        
        layout.addWidget(class_group)
        
        # Processing Settings
        process_group = QGroupBox("Image Processing Settings")
        process_layout = QFormLayout(process_group)
        
        self.image_size = QSpinBox()
        self.image_size.setRange(128, 512)
        self.image_size.setValue(224)
        process_layout.addRow("Standard Image Size:", self.image_size)
        
        self.preprocessing = QComboBox()
        self.preprocessing.addItems(["None", "Gaussian Blur", "Median Filter", "Bilateral Filter"])
        process_layout.addRow("Preprocessing:", self.preprocessing)
        
        layout.addWidget(process_group)
        
        # Model Performance Info (Removed, replaced with General Info)
        perf_group = QGroupBox("General Information")
        perf_layout = QVBoxLayout(perf_group)
        
        info_text = """
        This version of the application performs liquid textile waste detection
        using manual image processing and feature extraction, without
        relying on machine learning classification models.
        
        Detection is primarily based on:
        • Color, Shape, and Texture feature extraction.
        • Contour detection and bounding box visualization.
        
        This is suitable for visual inspection and understanding the
        underlying image processing steps.
        """
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("border: none; background-color: #34495e; padding: 10px;")
        perf_layout.addWidget(info_label)
        
        layout.addWidget(perf_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        apply_btn = QPushButton("Apply Settings")
        apply_btn.clicked.connect(self.apply_settings)
        button_layout.addWidget(apply_btn)
        
        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self.reset_settings)
        button_layout.addWidget(reset_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        # Simpan referensi ke main window
        self.parent().settings_dialog_ref = self
        # Sembunyikan/Show Masking Color Selection sesuai pilihan
        self.feature_type_combo.currentIndexChanged.connect(self.parent().update_masking_color_visibility)
    
    def apply_settings(self):
        """Apply current settings"""
        QMessageBox.information(self, "Settings", "Settings applied successfully!")
    
    def reset_settings(self):
        """Reset to default settings"""
        self.color_bins.setValue(32)
        self.lbp_radius.setValue(8)
        self.lbp_points.setValue(24)
        self.confidence_threshold.setValue(0.7) # Re-purposed
        self.image_size.setValue(224)
        self.preprocessing.setCurrentIndex(0)


# Removed DatasetGeneratorDialog

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application icon and info
    app.setApplicationName("Liquid Textile Waste Detection System (Manual Edition)")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Professional Edition")
    
    # Create main window
    window = TextileWasteDetector()
    
    # Add menu bar
    menubar = window.menuBar()
    menubar.setStyleSheet("""
        QMenuBar {
            background-color: #2c3e50;
            color: white;
            border-bottom: 1px solid #3498db;
        }
        QMenuBar::item {
            padding: 8px 12px;
        }
        QMenuBar::item:selected {
            background-color: #3498db;
        }
        QMenu {
            background-color: #34495e;
            color: white;
            border: 1px solid #3498db;
        }
        QMenu::item:selected {
            background-color: #3498db;
        }
    """)
    
    # File menu
    file_menu = menubar.addMenu('File')
    
    load_image_action = QAction('Load Image...', window)
    load_image_action.setShortcut('Ctrl+O')
    load_image_action.triggered.connect(window.browse_image)
    file_menu.addAction(load_image_action)
    
    file_menu.addSeparator()
    
    save_results_action = QAction('Save Results...', window)
    save_results_action.setShortcut('Ctrl+S')
    # Connect to a placeholder or simplified save if actual ML results saving is removed
    # For now, we keep the action but it might need more specific implementation if user wants to save processed images
    file_menu.addAction(save_results_action) 
    
    file_menu.addSeparator()
    
    exit_action = QAction('Exit', window)
    exit_action.setShortcut('Ctrl+Q')
    exit_action.triggered.connect(window.close)
    file_menu.addAction(exit_action)
    
    # Tools menu
    tools_menu = menubar.addMenu('Tools')
    
    # dataset_gen_action removed
    # tools_menu.addAction(dataset_gen_action)
    
    settings_action = QAction('Settings...', window)
    settings_action.triggered.connect(window.open_settings)
    tools_menu.addAction(settings_action)
    
    # Help menu
    help_menu = menubar.addMenu('Help')
    
    about_action = QAction('About', window)
    about_action.triggered.connect(lambda: QMessageBox.about(
        window, "About", 
        "Liquid Textile Waste Detection System v1.0 (Manual Edition)\n\n"
        "Professional image analysis tool for detecting\n"
        "and visualizing liquid textile waste using advanced\n"
        "computer vision techniques (without machine learning classification).\n\n"
        "Features:\n"
        "• Color, Shape, and Texture feature extraction\n"
        "• Real-time visual detection and highlighting\n"
        "• Professional GUI interface"
    ))
    help_menu.addAction(about_action)
    
    # Show window
    window.show()
    
    # Start application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main() 