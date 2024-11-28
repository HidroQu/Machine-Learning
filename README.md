# Nutrient and Plant Detection API

Projek ini bertujuan untuk mendeteksi jenis tanaman dan kekurangan nutrisi pada tanaman berdasarkan gambar yang diunggah pengguna. Proyek ini menggunakan framework Flask dengan model Machine Learning yang dilatih menggunakan TensorFlow dan data disimpan di database MySQL.

## Table of Contents
- [Fitur](#fitur)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Persiapan](#persiapan)
- [Endpoint API](#endpoint-api)
- [Cara Penggunaan](#cara-penggunaan)
- [Kontributor](#kontributor)
- [Lisensi](#lisensi)

---

## Fitur
- Prediksi kekurangan nutrisi tanaman berdasarkan gambar daun.
- Prediksi jenis tanaman berdasarkan gambar.
- Integrasi dengan MySQL untuk mendapatkan informasi tambahan terkait hasil prediksi.

## Teknologi yang Digunakan
- **Framework**: Flask
- **Model Machine Learning**: TensorFlow/Keras
- **Database**: MySQL
- **Bahasa Pemrograman**: Python

## Persiapan
### Prasyarat
Pastikan Anda sudah menginstal:
- Python (versi 3.8 atau lebih baru)
- MySQL Server
- Dependensi yang tercantum dalam `requirements.txt`

### Instalasi
1. Clone repository ini:
   ```bash
   git clone https://github.com/username/repository.git
   cd repository

2. Buat virtual environment dan aktifkan:
  python -m venv venv
  source venv/bin/activate  # Untuk Linux/MacOS
  venv\Scripts\activate     # Untuk Windows

3 Install dependensi:
  pip install -r requirements.txt

4. Jalankan aplikasi:
  python API/app.py

