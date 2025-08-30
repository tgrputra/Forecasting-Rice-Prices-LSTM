-----

# Prediksi Harga Beras di Wonosobo Menggunakan Metode LSTM

Aplikasi web ini menyajikan sistem prediksi harga untuk komoditas beras kualitas premium dan medium di Kabupaten Wonosobo, Jawa Tengah. Dibuat sebagai bagian dari Laporan Tugas Akhir, aplikasi ini menggunakan model *Deep Learning* **Long Short-Term Memory (LSTM)** untuk meramalkan pergerakan harga di masa depan dan melakukan validasi terhadap data historis.

Aplikasi ini dibangun menggunakan **Streamlit** dan menyajikan tiga fitur utama untuk analisis harga beras.

## Fitur-fitur 📋

1.  **Prediksi Maju**: Memprediksi pergerakan harga beras untuk periode 1 hingga 180 hari ke depan. Pengguna dapat memilih tanggal akhir prediksi melalui kalender untuk mendapatkan visualisasi dan tabel hasil proyeksi harga.
2.  **Prediksi Mundur (Validasi Historis)**: Menguji keandalan model dengan memprediksi harga pada rentang tanggal di masa lalu. Pengguna dapat memilih rentang hingga 180 hari untuk melihat perbandingan antara harga aktual dan hasil prediksi model.
3.  **Cek Harga Historis**: Fitur untuk melihat data harga aktual pada rentang tanggal tertentu (maksimal 1 bulan) dalam format tabel.

## Tampilan Aplikasi 🖼️

  - **Antarmuka Utama**: Menampilkan grafik interaktif dari data historis dan hasil prediksi.
  - **Sidebar**: Berisi semua kontrol untuk menjalankan fitur-fitur prediksi, yang dipisahkan dalam tab **"Prediksi Maju"** dan **"Prediksi Mundur"**.
  - **Ringkasan**: Kartu metrik menampilkan ringkasan hasil prediksi, seperti harga tertinggi, terendah, dan tren harga.

## Teknologi yang Digunakan 💻

  * **Bahasa Pemrograman**: Python
  * **Framework Aplikasi Web**: Streamlit
  * **Model Deep Learning**: TensorFlow & Keras
  * **Analisis Data**: Pandas & NumPy
  * **Visualisasi**: Plotly

## Cara Menjalankan Aplikasi

1.  Pastikan Anda memiliki Python dan `pip` terinstal.
2.  *Clone* repositori ini.
3.  Instal semua *dependency* yang dibutuhkan:
    ```bash
    pip install -r requirements.txt
    ```
4.  Jalankan aplikasi Streamlit dari terminal:
    ```bash
    streamlit run app.py
    ```
5. Atau langsung akses ke link berikut https://forecasting-rice-prices-lstm.streamlit.app/
## Struktur Proyek

```
.
├── Model/
│   ├── lstm_model_premium.ipynb
│   └── lstm_model_medium.ipynb
├── Prepocessing/
│   └── data_preprocessing.ipynb
├── app.py                      # Skrip utama aplikasi Streamlit
├── dataset_beras_lstm.csv      # Dataset yang digunakan
├── model_prediksi_beras_*.h5   # File model terlatih
├── scaler_*.pkl                # File scaler untuk normalisasi
└── requirements.txt            # Daftar dependency
```
