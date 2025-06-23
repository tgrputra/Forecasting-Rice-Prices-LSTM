import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from datetime import timedelta

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(
    page_title="Prediksi Harga Beras",
    page_icon="🌾",
    layout="wide"
)
st.title("🌾Aplikasi Prediksi Harga Komoditas Beras Wonosobo")
st.markdown("Aplikasi interaktif untuk visualisasi dan prediksi harga beras kualitas premium dan medium.")

# --- Fungsi-fungsi Bantuan (dengan cache untuk performa) ---

@st.cache_resource
def load_model_and_scaler(jenis_beras):
    """Memuat model dan scaler berdasarkan jenis beras yang dipilih."""
    try:
        if jenis_beras == 'Premium':
            model = load_model('model_prediksi_beras_premium.h5')
            with open('scaler_premium.pkl', 'rb') as f:
                scaler = pickle.load(f)
        else: # Medium
            model = load_model('model_prediksi_beras_medium.h5')
            with open('scaler_medium.pkl', 'rb') as f:
                scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_data():
    """Memuat dan memproses dataset dari file CSV."""
    try:
        df = pd.read_csv('Dataset Beras New Fix.csv')
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df.set_index('Tanggal', inplace=True)
        return df
    except FileNotFoundError:
        return None

def predict_future(model, scaler, historical_data, window_size, n_future):
    """Fungsi untuk melakukan prediksi iteratif ke masa depan."""
    scaled_data = scaler.transform(historical_data.values.reshape(-1, 1))
    last_sequence = scaled_data[-window_size:]
    current_batch = last_sequence.reshape(1, 1, window_size)
    future_predictions_scaled = []
    
    for _ in range(n_future):
        next_pred_scaled = model.predict(current_batch, verbose=0)[0]
        future_predictions_scaled.append(next_pred_scaled)
        reshaped_pred = next_pred_scaled.reshape(1, 1, 1)
        current_batch = np.append(current_batch[:, :, 1:], reshaped_pred, axis=2)
        
    future_predictions = scaler.inverse_transform(future_predictions_scaled)
    return future_predictions

@st.cache_data
def convert_df_to_csv(df):
    """Mengubah DataFrame menjadi CSV untuk di-download."""
    return df.to_csv(index=False).encode('utf-8')

# --- Memuat Data Utama ---
data = load_data()

if data is None:
    st.error("File 'Dataset Beras New Fix.csv' tidak ditemukan. Mohon letakkan file di folder yang sama.")
else:
    # --- Sidebar untuk Pengaturan ---
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        
        jenis_beras = st.selectbox('Pilih Jenis Beras:', ('Premium', 'Medium'))
        
        kolom_data = data[[jenis_beras]].dropna()
        model, scaler = load_model_and_scaler(jenis_beras)
        
        if model is None or scaler is None:
            st.error(f"File model/scaler untuk beras {jenis_beras} tidak ditemukan.")
            st.stop()

        rentang_waktu = st.selectbox(
            "Pilih Rentang Waktu Historis:",
            ('3 Bulan Terakhir', '6 Bulan Terakhir', '1 Tahun Terakhir', 'Semua Data')
        )
        
        n_future = st.slider('Pilih Jumlah Hari Prediksi:', min_value=1, max_value=30, value=7)
        start_prediction = st.button('Mulai Prediksi', type="primary", use_container_width=True)
        
        st.markdown("---")
        st.info(
            "**Tentang Model:**\n"
            "Prediksi ini dibuat menggunakan model time-series LSTM (Long Short-Term Memory). "
            "Akurasi model dievaluasi menggunakan RMSE dan MAPE pada data validasi."
        )

    # Filter data berdasarkan rentang waktu yang dipilih
    today = kolom_data.index.max()
    if rentang_waktu == '3 Bulan Terakhir':
        data_to_plot = kolom_data[today - pd.DateOffset(months=3):]
    elif rentang_waktu == '6 Bulan Terakhir':
        data_to_plot = kolom_data[today - pd.DateOffset(months=6):]
    elif rentang_waktu == '1 Tahun Terakhir':
        data_to_plot = kolom_data[today - pd.DateOffset(years=1):]
    else:
        data_to_plot = kolom_data

    # --- Tampilan Utama ---
    st.header(f"Visualisasi dan Prediksi Harga Beras {jenis_beras}")

    # Inisialisasi grafik Plotly
    fig = go.Figure()

    # Tambahkan trace data historis
    fig.add_trace(go.Scatter(x=data_to_plot.index, y=data_to_plot[jenis_beras], mode='lines', name='Data Historis', line=dict(color='blue')))

    if start_prediction:
        with st.spinner(f'Melakukan prediksi untuk {n_future} hari ke depan...'):
            window_size = 20
            predictions = predict_future(model, scaler, kolom_data, window_size, n_future)
            
            last_date = kolom_data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_future)
            
            # --- Tampilkan Kartu Metrik ---
            st.subheader("Ringkasan Prediksi")
            col1, col2, col3, col4 = st.columns(4)
            
            harga_besok = predictions[0][0]
            col1.metric("Prediksi Harga Besok", f"Rp {harga_besok:,.0f}")

            harga_terendah = predictions.min()
            col2.metric("Harga Terendah", f"Rp {harga_terendah:,.0f}")
            
            harga_tertinggi = predictions.max()
            col3.metric("Harga Tertinggi", f"Rp {harga_tertinggi:,.0f}")
            
            tren = "Naik 📈" if predictions[-1] > kolom_data[jenis_beras].iloc[-1] else "Turun 📉"
            delta_tren = f"{abs(predictions[-1][0] - kolom_data[jenis_beras].iloc[-1]):,.0f}"
            col4.metric("Tren Prediksi", tren, delta=delta_tren)

            # Buat DataFrame hasil prediksi
            df_pred = pd.DataFrame({
                'Tanggal': future_dates,
                'Harga Prediksi (Rp)': predictions.flatten()
            })
            df_pred['Tanggal'] = df_pred['Tanggal'].dt.strftime('%Y-%m-%d')

            # Tambahkan trace hasil prediksi ke grafik
            fig.add_trace(go.Scatter(x=df_pred['Tanggal'], y=df_pred['Harga Prediksi (Rp)'], mode='lines', name='Hasil Prediksi', line=dict(color='red', dash='dash')))
            
            # Tambahkan area yang diarsir untuk prediksi
            fig.add_trace(go.Scatter(
                x=np.concatenate([data_to_plot.index[-1:], df_pred['Tanggal']]),
                y=np.concatenate([data_to_plot[jenis_beras][-1:], df_pred['Harga Prediksi (Rp)']]),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.1)',
                mode='none',
                name='Area Prediksi',
                showlegend=False
            ))

            # Tampilkan tabel prediksi
            st.subheader(f"Tabel Hasil Prediksi Harga")
            st.dataframe(df_pred.style.format({"Harga Prediksi (Rp)": "Rp {:,.0f}"}), use_container_width=True, hide_index=True)

            # Tombol Download
            csv_pred = convert_df_to_csv(df_pred)
            st.download_button(
               label="📥 Download Hasil Prediksi (CSV)",
               data=csv_pred,
               file_name=f'prediksi_beras_{jenis_beras.lower()}_{n_future}_hari.csv',
               mime='text/csv',
            )
            st.success("Prediksi berhasil dibuat!")


    # Update layout grafik
    fig.update_layout(
        title=f'Grafik Harga Beras {jenis_beras}',
        xaxis_title='Tanggal',
        yaxis_title='Harga (Rp)',
        legend_title='Keterangan',
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Expander untuk Analisis Historis ---
    with st.expander("📊 Lihat Analisis Data Historis (1 Tahun Terakhir)"):
        data_1_tahun = data[today - pd.DateOffset(years=1):]
        harga_rata_rata = data_1_tahun[jenis_beras].mean()
        harga_min_hist = data_1_tahun[jenis_beras].min()
        harga_max_hist = data_1_tahun[jenis_beras].max()
        
        col_hist1, col_hist2, col_hist3 = st.columns(3)
        col_hist1.metric("Harga Rata-rata (1 Thn)", f"Rp {harga_rata_rata:,.0f}")
        col_hist2.metric("Harga Terendah (1 Thn)", f"Rp {harga_min_hist:,.0f}")
        col_hist3.metric("Harga Tertinggi (1 Thn)", f"Rp {harga_max_hist:,.0f}")