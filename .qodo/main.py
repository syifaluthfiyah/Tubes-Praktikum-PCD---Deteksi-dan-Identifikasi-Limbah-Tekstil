import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt

# Import dari file image_processing.py yang baru
from image_processing import FeatureExtractor, detect_jeans_shape, detect_texture_type, process_image_for_display

class TextileWasteDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.current_image = None
        self.processed_image = None
        self.selected_feature_type = "Warna (Limbah Cair)"  # Default
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Waste Detection System - Manual Edition')
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
        self.statusBar().showMessage('System Ready - Select an image to begin detection')
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
        
        self.detect_btn = QPushButton("🔍 Start Detection")
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
        self.tab_widget.addTab(self.detection_tab, "🔍Detection Result")
        
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
        result_group = QGroupBox("❌ Detection Result")
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
    
    def start_detection(self):
        if not hasattr(self, 'current_image_bgr') or self.current_image_bgr is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
        try:
            self.log_message("Starting detection process (manual feature extraction)...")
            mode = "cair" if self.selected_feature_type == "Warna (Limbah Cair)" else (
                "bentuk" if self.selected_feature_type == "Bentuk (Limbah Padat)" else "tekstur")
            
            # Panggil fungsi process_image_for_display dari image_processing.py
            self.processed_image, self.images_for_combined_plot = process_image_for_display(
                self.current_image_bgr.copy(), 
                mode=mode,
                feature_extractor=self.feature_extractor,
                mask_color_combo_currentText=self.mask_color_combo.currentText()
            )
            
            # Tentukan subfolder berdasarkan mode deteksi
            if mode == "cair":
                output_subfolder = "LimbahCair"
            elif mode == "bentuk":
                output_subfolder = "Jeans"
            else: # mode == "tekstur"
                output_subfolder = "Plastik-jeans"

            # Simpan gambar hasil akhir ke folder Output yang spesifik
            base_output_dir = "./Output"
            final_output_dir = os.path.join(base_output_dir, output_subfolder)
            os.makedirs(final_output_dir, exist_ok=True)
            file_name = os.path.basename(self.image_status.text().replace("Image loaded: ", ""))
            output_path = os.path.join(final_output_dir, f"result_{file_name}")
            cv2.imwrite(output_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))
            self.log_message(f"Final detection result saved to: {output_path}")
            
            self.update_detection_results_manual()
            self.log_message("Detection completed successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")
            self.log_message(f"Detection error: {str(e)}")
    
    def update_detection_results_manual(self):
        """Update detection results in UI untuk multi-preview processed image"""
        # Tentukan mode aktif
        mode = "cair" if self.selected_feature_type == "Warna (Limbah Cair)" else (
            "bentuk" if self.selected_feature_type == "Bentuk (Limbah Padat)" else "tekstur")
        # Update judul group box dan summary/statistik sesuai mode
        if mode == "cair":
            self.detection_tab.findChild(QGroupBox, None).setTitle("🎯 Liquid Detection Summary")
            self.summary_label.setText("Detection Results:\nVisual detection performed focusing on liquid textile waste areas.\n")
            self.stats_label.setText("Analysis based on HSV color model:\nFeatures Extracted: Multi-Color Liquid Waste Detection")
        elif mode == "bentuk":
            self.detection_tab.findChild(QGroupBox, None).setTitle("🎯 Shape Detection Summary")
            self.summary_label.setText("Detection Results:\nVisual detection performed focusing on jeans shape analysis (good/damaged/not jeans).\n")
            self.stats_label.setText("Analysis based on shape features:\nFeatures Extracted: Aspect Ratio, Extent, Solidity, Symmetry")
        else:  # tekstur
            self.detection_tab.findChild(QGroupBox, None).setTitle("🎯 Texture Detection Summary")
            self.summary_label.setText("Detection Results:\nVisual detection performed focusing on textile vs plastic texture analysis.\n")
            self.stats_label.setText("Analysis based on texture features:\nFeatures Extracted: GLCM, LBP, Rule-based Classification")
        # Tampilkan hasil deteksi (bounding box) sesuai mode di result_label
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
        
        if not hasattr(self, 'images_for_combined_plot') or not self.images_for_combined_plot:
            return
        
        # Tentukan subfolder berdasarkan mode deteksi
        mode = "cair" if self.selected_feature_type == "Warna (Limbah Cair)" else (
            "bentuk" if self.selected_feature_type == "Bentuk (Limbah Padat)" else "tekstur")

        if mode == "cair":
            processing_subfolder = "LimbahCair"
        elif mode == "bentuk":
            processing_subfolder = "Jeans"
        else: # mode == "tekstur"
            processing_subfolder = "Plastik-jeans"

        # Buat plot gabungan menggunakan Matplotlib
        n_images = len(self.images_for_combined_plot)
        fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
        if n_images == 1: # Handle single image case
            axes = [axes]

        for i, (img, title) in enumerate(self.images_for_combined_plot):
            axes[i].imshow(img)
            axes[i].set_title(title, color='white')
            axes[i].axis('off')

        plt.tight_layout()
        
        # Simpan plot gabungan ke folder ProcessingImage yang spesifik
        base_processing_dir = "./ProcessingImage"
        final_processing_dir = os.path.join(base_processing_dir, processing_subfolder)
        os.makedirs(final_processing_dir, exist_ok=True)
        file_name = os.path.basename(self.image_status.text().replace("Image loaded: ", ""))
        plot_path = os.path.join(final_processing_dir, f"processed_steps_{file_name}.png")
        plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor='none') # Simpan dengan background gelap
        self.log_message(f"Combined processing images saved to: {plot_path}")
        
        plt.close(fig) # Tutup figure untuk menghemat memori

        # Tampilkan gambar gabungan di GUI (bukan sebagai QLabel terpisah lagi, tapi sebagai QPixmap dari file yang disimpan)
        try:
            combined_image_pixmap = QPixmap(plot_path)
            if not combined_image_pixmap.isNull():
                # Buat QLabel baru untuk menampung gambar gabungan
                combined_label = QLabel()
                combined_label.setAlignment(Qt.AlignCenter)
                
                # Skala gambar agar sesuai dengan area yang tersedia di tab processed_tab
                label_width = self.processed_tab.width()
                label_height = self.processed_tab.height()
                scaled_pixmap = combined_image_pixmap.scaled(label_width, label_height, 
                                                             Qt.KeepAspectRatio, Qt.SmoothTransformation)
                combined_label.setPixmap(scaled_pixmap)
                self.processed_images_layout.addWidget(combined_label)
            else:
                self.log_message(f"Error: Could not load combined image from {plot_path}")
        except Exception as e:
            self.log_message(f"Error displaying combined image: {str(e)}")
    
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
                # Karena process_image_for_display sekarang mengembalikan hasil lengkap,
                # kita perlu mengadaptasi bagian ini.
                # Untuk saat ini, kita akan membuat dummy data atau mengambil dari hasil terakhir jika memungkinkan.
                self.analysis_table.setRowCount(1) # Asumsi hanya satu objek utama
                self.analysis_table.setItem(0, 0, QTableWidgetItem("Objek_1"))
                self.analysis_table.setItem(0, 1, QTableWidgetItem("-"))
                # Karena tidak ada jeans_result global lagi, ini perlu disesuaikan jika ingin detail per objek.
                # Untuk sementara, akan menampilkan 'Terdeteksi' atau 'Tidak terdeteksi'
                self.analysis_table.setItem(0, 2, QTableWidgetItem("Terdeteksi (Bentuk)")) 
                self.analysis_table.setItem(0, 3, QTableWidgetItem("N/A"))
                self.analysis_table.setItem(0, 4, QTableWidgetItem("N/A"))
                self.analysis_table.setItem(0, 5, QTableWidgetItem("N/A"))
            else:  # Tekstur
                # Sama seperti mode bentuk, perlu disesuaikan.
                self.analysis_table.setRowCount(1) # Asumsi hanya satu objek utama
                self.analysis_table.setItem(0, 0, QTableWidgetItem("Objek_1"))
                self.analysis_table.setItem(0, 1, QTableWidgetItem("-"))
                self.analysis_table.setItem(0, 2, QTableWidgetItem("Terdeteksi (Tekstur)"))
                self.analysis_table.setItem(0, 3, QTableWidgetItem("N/A"))
                self.analysis_table.setItem(0, 4, QTableWidgetItem("N/A"))
                self.analysis_table.setItem(0, 5, QTableWidgetItem("N/A"))
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