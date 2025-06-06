import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QPushButton,
                            QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit, QFrame, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from PIL import Image
import os

class CardWidget(QFrame):
    """Custom card-style widget for image display."""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #f9fafd;
                border-radius: 16px;
                border: 1.5px solid #e0e0e0;
                box-shadow: 0px 2px 8px rgba(0,0,0,0.07);
            }
        """)
        self.setFixedSize(270, 320)
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 15px; color: #2d3a4a; margin-bottom: 8px;")
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(250, 250)
        self.image_label.setStyleSheet("background-color: #f4f6fa; border: 1px solid #e0e0e0; border-radius: 10px;")
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #0078D7; margin-top: 8px;")
        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.result_label)
        self.setLayout(self.layout)

    def set_image(self, image):
        self.image_label.setPixmap(image)

    def set_result(self, text, color="#0078D7"):
        self.result_label.setText(text)
        self.result_label.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {color}; margin-top: 8px;")

class LimbahDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteksi Limbah Cair - RGB Feature Extraction")
        self.setGeometry(100, 100, 1280, 900)
        self.setStyleSheet("""
            QWidget {
                background-color: #eaf1fb;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
        """)
        self.image_path = None

        # Main Layout
        self.layout = QVBoxLayout()
        self.layout.setSpacing(18)
        self.layout.setContentsMargins(32, 24, 32, 24)

        # Header
        header_layout = QHBoxLayout()
        icon_path = os.path.join(os.path.dirname(__file__), "water-drop.png")
        if os.path.exists(icon_path):
            icon_label = QLabel()
            icon_label.setPixmap(QPixmap(icon_path).scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            header_layout.addWidget(icon_label)
        title = QLabel("Deteksi Limbah Cair - RGB Feature Extraction")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #1a2636; padding: 8px 0 8px 12px;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        self.layout.addLayout(header_layout)

        # Button
        self.load_button = QPushButton("  Load Image")
        self.load_button.setIcon(QIcon.fromTheme("document-open"))
        self.load_button.clicked.connect(self.load_image)
        self.load_button.setCursor(Qt.PointingHandCursor)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                font-weight: bold;
                font-size: 15px;
                padding: 10px 28px;
                border-radius: 8px;
                border: none;
                margin-bottom: 10px;
            }
            QPushButton:hover {
                background-color: #005a9e;
                box-shadow: 0px 2px 8px rgba(0,0,0,0.10);
            }
        """)
        self.layout.addWidget(self.load_button, alignment=Qt.AlignLeft)

        # Image Cards
        self.image_cards_layout = QHBoxLayout()
        self.image_cards_layout.setSpacing(18)
        self.original_card = CardWidget("Original Image")
        self.color_card = CardWidget("Color Feature")
        self.shape_card = CardWidget("Shape Feature")
        self.texture_card = CardWidget("Texture Feature")
        for card in [self.original_card, self.color_card, self.shape_card, self.texture_card]:
            self.image_cards_layout.addWidget(card)
        self.layout.addLayout(self.image_cards_layout)

        # Console Title
        console_title = QLabel("Console Output")
        console_title.setStyleSheet("font-weight: bold; font-size: 15px; color: #1a2636; margin-top: 18px;")
        self.layout.addWidget(console_title)

        # Console Output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("""
            QTextEdit {
                background-color: #23272e;
                color: #e0e0e0;
                font-family: 'Consolas', 'Fira Mono', 'monospace';
                font-size: 13px;
                border-radius: 8px;
                border: 1.5px solid #1a2636;
                padding: 10px;
                min-height: 120px;
            }
        """)
        self.layout.addWidget(self.console)

        self.setLayout(self.layout)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.image_path = file_name
            image = cv2.imread(file_name)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.display_image(image_rgb, self.original_card)
            self.console.append(f"<span style='color:#7ecfff;'>[INFO]</span> Gambar dimuat: {file_name}")

            self.process_color_feature(image_rgb)
            self.process_shape_feature(image_rgb)
            self.process_texture_feature(image_rgb)

    def display_image(self, image, card):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        card.set_image(pixmap)

    def process_color_feature(self, image):
        img_reshape = image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=2, random_state=42).fit(img_reshape)
        clustered = kmeans.labels_.reshape(image.shape[:2])
        mean_color = np.mean(kmeans.cluster_centers_, axis=0)
        color_image = np.zeros_like(image)
        color_image[clustered == 1] = (255, 255, 255)
        self.display_image(color_image, self.color_card)

        if np.linalg.norm(mean_color - [0, 0, 255]) > 50:
            result = "Limbah terdeteksi"
            color = "#e74c3c"
        else:
            result = "Limbah tidak terdeteksi"
            color = "#27ae60"

        self.color_card.set_result("Color: " + result, color)
        self.console.append(f"<span style='color:#f7ca18;'>[COLOR]</span> Rata-rata warna: {mean_color}, Hasil: <b>{result}</b>")

    def process_shape_feature(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_image = cv2.drawContours(np.zeros_like(image), contours, -1, (255, 255, 255), 2)
        self.display_image(shape_image, self.shape_card)

        if len(contours) > 5:
            result = "Limbah terdeteksi"
            color = "#e74c3c"
        else:
            result = "Limbah tidak terdeteksi"
            color = "#27ae60"

        self.shape_card.set_result("Shape: " + result, color)
        self.console.append(f"<span style='color:#f9bf3b;'>[SHAPE]</span> Jumlah kontur: {len(contours)}, Hasil: <b>{result}</b>")

    def process_texture_feature(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

        (hist, _)= np.histogram(lbp.ravel(),
                                bins=np.arange(0, n_points + 3),
                                range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)

        uniformity = np.sum(hist[1:-1])
        lbp_vis = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min()))
        lbp_rgb = np.stack([lbp_vis] * 3, axis=-1)

        self.display_image(lbp_rgb, self.texture_card)

        if uniformity > 0.5:
            result = "Limbah terdeteksi"
            color = "#e74c3c"
        else:
            result = "Limbah tidak terdeteksi"
            color = "#27ae60"

        self.texture_card.set_result("Texture: " + result, color)
        self.console.append(f"<span style='color:#7ed6df;'>[TEXTURE]</span> Uniformitas LBP: {uniformity:.4f}, Hasil: <b>{result}</b>")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LimbahDetector()
    window.show()
    sys.exit(app.exec_())