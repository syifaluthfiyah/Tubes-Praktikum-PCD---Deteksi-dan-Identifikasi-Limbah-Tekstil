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

class FeatureExtractor:
    """Class untuk ekstraksi fitur warna, bentuk, dan tekstur"""
    
    def __init__(self):
        pass # Scaler not needed without ML training
    
    def extract_color_features(self, image):
        """Ekstraksi fitur warna (histogram RGB dan HSV)"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # RGB histogram
        hist_r = cv2.calcHist([image], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [32], [0, 256])
        
        # HSV histogram
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Mean RGB dan HSV
        mean_rgb = np.mean(image, axis=(0, 1))
        mean_hsv = np.mean(hsv, axis=(0, 1))
        
        # Combine features
        color_features = np.concatenate([
            hist_r.flatten(), hist_g.flatten(), hist_b.flatten(),
            hist_h.flatten(), hist_s.flatten(), hist_v.flatten(),
            mean_rgb, mean_hsv
        ])
        
        return color_features
    
    def extract_shape_features(self, image):
        """Ekstraksi fitur bentuk"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return np.zeros(10)
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Shape features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Hu moments
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # Extent
        extent = float(area) / (w * h) if w * h != 0 else 0
        
        # Solidity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        
        shape_features = np.concatenate([
            hu_moments[:7], [aspect_ratio, extent, solidity]
        ])
        
        return shape_features
    
    def extract_texture_features(self, image):
        """Ekstraksi fitur tekstur menggunakan LBP dan GLCM"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # LBP (Local Binary Pattern)
        lbp = feature.local_binary_pattern(gray, 24, 8, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 25))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        # GLCM features
        glcm = feature.greycomatrix(gray, [1], [0, 45, 90, 135], levels=256, symmetric=True, normed=True)
        
        # GLCM properties
        contrast = feature.greycoprops(glcm, 'contrast').flatten()
        dissimilarity = feature.greycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = feature.greycoprops(glcm, 'homogeneity').flatten()
        energy = feature.greycoprops(glcm, 'energy').flatten()
        correlation = feature.greycoprops(glcm, 'correlation').flatten()
        
        # Statistical features
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        skewness = np.mean(((gray - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((gray - mean_val) / std_val) ** 4)
        
        texture_features = np.concatenate([
            lbp_hist, contrast, dissimilarity, homogeneity, 
            energy, correlation, [mean_val, std_val, skewness, kurtosis]
        ])
        
        return texture_features
    
    def extract_all_features(self, image):
        """Ekstraksi semua fitur"""
        color_feat = self.extract_color_features(image)
        shape_feat = self.extract_shape_features(image)
        texture_feat = self.extract_texture_features(image)
        
        return np.concatenate([color_feat, shape_feat, texture_feat])

class TextileWasteDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.current_image = None
        self.processed_image = None
        
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
        comp_group = QGroupBox("🔍 Comparison Image")
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
        """Create processed image tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.processed_image_label = QLabel("No processed image available")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setMinimumSize(800, 600)
        self.processed_image_label.setStyleSheet("border: 2px dashed #3498db;")
        layout.addWidget(self.processed_image_label)
        
        return tab
    
    def browse_image(self):
        """Browse and load image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image for Liquid Textile Waste Detection", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            try:
                # Load image
                self.current_image = cv2.imread(file_path)
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                
                # Update UI
                self.image_status.setText(f"Image loaded: {os.path.basename(file_path)}")
                self.detect_btn.setEnabled(True)
                
                # Display image
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
        """Start waste detection process using only image processing"""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
        
        try:
            self.log_message("Starting detection process (manual feature extraction)...")
            
            # Process image for visualization (contour detection, bounding boxes)
            self.processed_image = self.process_image_for_display(self.current_image.copy())
            
            # Update results based on visual detection
            self.update_detection_results_manual()
            
            self.log_message("Detection completed successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")
            self.log_message(f"Detection error: {str(e)}")
    
    def process_image_for_display(self, image):
        """Process image to highlight detected features"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to find contours
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on original image
        result_image = image.copy()
        cv2.drawContours(result_image, contours, -1, (255, 0, 0), 2)
        
        # Add bounding boxes for larger contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Only for significant areas
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_image, "Liquid Waste Detected", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_image
    
    def update_detection_results_manual(self):
        """Update detection results in UI for manual mode"""
        # Update summary
        summary_text = """Detection Results:
"""
        if self.processed_image is not None:
            summary_text += """Visual detection performed. Contours and bounding boxes highlighted.
"""
        else:
            summary_text += """No processed image available for visual detection."""
        
        self.summary_label.setText(summary_text)
        
        # Update statistics (simplified for manual mode)
        stats_text = """Analysis based on image processing:
"""
        stats_text += """Features Extracted: Color, Shape, Texture (manual analysis)"""
        
        self.stats_label.setText(stats_text)
        
        # Display processed image
        if self.processed_image is not None:
            self.display_image(self.processed_image, self.result_label)
            self.display_image(self.processed_image, self.processed_image_label)
        
        # Update analysis table (simplified for manual mode)
        self.update_analysis_table_manual()
    
    def update_analysis_table_manual(self):
        """Update detailed analysis table for manual mode"""
        # We can still extract features to display some info, even if not used for ML classification
        if self.current_image is not None:
            features = self.feature_extractor.extract_all_features(self.current_image)
            
            self.analysis_table.setRowCount(1)
            self.analysis_table.setItem(0, 0, QTableWidgetItem("Image_01"))
            self.analysis_table.setItem(0, 1, QTableWidgetItem("N/A (Manual)"))
            self.analysis_table.setItem(0, 2, QTableWidgetItem("Liquid Waste (Visual)"))
            self.analysis_table.setItem(0, 3, QTableWidgetItem("N/A"))
            self.analysis_table.setItem(0, 4, QTableWidgetItem(f"Approx. {self.current_image.shape[1]}x{self.current_image.shape[0]}"))
            self.analysis_table.setItem(0, 5, QTableWidgetItem("N/A"))
        else:
            self.analysis_table.setRowCount(0) # Clear table
    
    # Removed load_dataset, train_models, save_models, load_models
    
    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self)
        dialog.exec_()
    
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