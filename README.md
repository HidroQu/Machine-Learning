
# HidroQu - Machine Learning Path  

**Tagline:**  
*Empowering hydroponic farming with intelligent nutrient management and crop recognition.*  

## Deskripsi Proyek  
HidroQu adalah aplikasi yang mendukung petani hidroponik untuk memonitor kesehatan tanaman secara efisien. Melalui teknologi Machine Learning, HidroQu mampu mendeteksi defisiensi nutrisi seperti nitrogen, fosfor, dan kalium berdasarkan analisis visual daun. Selain itu, aplikasi ini juga memiliki fitur pengenalan tanaman untuk memastikan diagnosis dan rekomendasi yang akurat.  

## Fitur Utama  
1. **Pengenalan Tanaman:** Mendeteksi jenis tanaman hidroponik, termasuk bok choy, bayam, kale, selada, tomat, dan mentimun, dengan *Test Accuracy* mencapai **97.11%**.  
2. **Deteksi Defisiensi Nutrisi:** Mengidentifikasi kekurangan nutrisi berdasarkan ciri-ciri daun dengan *Test Accuracy* mencapai **88.57%**.  

## Dataset  
- **Sumber:** Dataset dikumpulkan dari berbagai sumber seperti Kaggle, Mendeley Data, dan Google.  
- **Preprocessing:** Data mengalami augmentasi dan normalisasi sederhana.  

## Model dan Framework  
- **Arsitektur Model:** Convolutional Neural Network (CNN) berbasis Transfer Learning menggunakan MobileNetV2.  
- **Framework dan Tools:** TensorFlow untuk pengembangan model dan Flask untuk membuat endpoint API.  

## Integrasi dengan Path Lain  
- Model diekspor dalam format *saved model* untuk digunakan oleh path lain.  
- Endpoint dibuat menggunakan Flask, yang terhubung dengan database melalui Cloud Computing path.  
- Endpoint ini memungkinkan integrasi langsung dengan aplikasi Android yang digunakan oleh pengguna.  

## Instruksi Penggunaan  
1. Clone repository ini:  
   ```bash
   git clone https://github.com/your-organization/your-repo.git
   cd your-repo
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Jalankan server Flask:  
   ```bash
   python app.py
   ```  
4. Untuk input dan output model, lihat contoh pada file Jupyter Notebook (`.ipynb`).  

## Hasil dan Evaluasi  
- **Pengenalan Tanaman:**  
  - Test Accuracy: **97.11%**  
  - Test Loss: **0.8313**  
- **Deteksi Defisiensi Nutrisi:**  
  - Accuracy in Test Data: **88.57%**  
  - Loss in Test Data: **47.30**  
- Visualisasi performa dapat dilihat pada grafik di file Jupyter Notebook.  

## Tim dan Kontribusi  
- **Muhammad Hanif Sya'bani**  
- **Rihan Naufaldihanif**  
- **Siti Latifah**  

## Lisensi  
Proyek ini menggunakan lisensi **MIT License**.  
