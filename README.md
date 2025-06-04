# 💧 Deteksi Limbah Cair Tercemar dan Tidak Tercemar

Proyek ini merupakan aplikasi berbasis **Python dengan GUI** yang bertujuan untuk melakukan **klasifikasi limbah cair** ke dalam dua kategori: **tercemar** dan **tidak tercemar**. Deteksi dilakukan melalui **ekstraksi fitur citra** menggunakan metode pengolahan citra digital.

## 🧠 Fitur Utama

- Ekstraksi Fitur Warna: histogram HSV, rata-rata RGB
- Ekstraksi Fitur Bentuk: kontur, luas, keliling, rasio aspek
- Ekstraksi Fitur Tekstur: GLCM, Haralick, atau LBP
- Antarmuka GUI: dibangun dengan `tkinter` atau `PyQt5`

## 📸 Cara Kerja Aplikasi

1. Pengguna memilih citra limbah cair melalui GUI.
2. Aplikasi melakukan preprocessing dan ekstraksi fitur:
   - Warna → Histogram HSV, Rata-rata RGB
   - Bentuk → Kontur dan geometri
   - Tekstur → GLCM, LBP, atau Haralick
3. Fitur digabung dan diklasifikasi dengan model.
4. Hasil klasifikasi ditampilkan: **Tercemar** atau **Tidak Tercemar**.
