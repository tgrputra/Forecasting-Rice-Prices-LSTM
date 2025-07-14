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
st.title("🌾 Aplikasi Prediksi Harga Komoditas Beras Wonosobo")
st.markdown("Aplikasi interaktif untuk visualisasi dan prediksi harga beras kualitas premium dan medium")

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
        df = pd.read_csv('dataset_beras_lstm.csv')
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
    df_to_download = df.copy()
    if 'Tanggal' in df_to_download.columns:
        df_to_download['Tanggal'] = pd.to_datetime(df_to_download['Tanggal']).dt.strftime('%Y-%m-%d')
    return df_to_download.to_csv(index=False).encode('utf-8')

# --- Memuat Data Utama ---
data = load_data()

if data is None:
    st.error("File 'dataset_beras_lstm.csv' tidak ditemukan. Mohon letakkan file di folder yang sama dengan app.py.")
else:
    # --- Sidebar untuk Pengaturan ---
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        
        jenis_beras = st.selectbox('1. Pilih Jenis Beras:', ('Premium', 'Medium'))
        
        kolom_data = data[[jenis_beras]].dropna()
        model, scaler = load_model_and_scaler(jenis_beras)
        
        if model is None or scaler is None:
            st.error(f"File model/scaler untuk beras {jenis_beras} tidak ditemukan.")
            st.stop()
        
        today = kolom_data.index.max()
        tab1, tab2 = st.tabs(["Prediksi Maju", "Prediksi Mundur"])

        with tab1:
            st.subheader("Pengaturan Prediksi Maju")
            rentang_waktu = st.selectbox(
                "Tampilan Rentang Grafik:",
                ('3 Bulan Terakhir', '6 Bulan Terakhir', '1 Tahun Terakhir', 'Semua Data'),
                key='pred_rentang'
            )
            
            min_pred_date = today.date() + pd.DateOffset(days=1)
            max_pred_date = today.date() + pd.DateOffset(days=180)

            selected_future_date = st.date_input(
                "Pilih Tanggal Prediksi:",
                value=today.date() + pd.DateOffset(days=30),
                min_value=min_pred_date,
                max_value=max_pred_date,
                help="Maksimal Prediksi 6 Bulan ke Depan.",
            )
            
            n_future = (selected_future_date - today.date()).days
            st.info(f"Rentang Prediksi Saat Ini: {n_future} Hari")
            
            start_prediction = st.button('Mulai Prediksi Maju', type="primary", use_container_width=True)

        with tab2:
            st.subheader("Pengaturan Prediksi Mundur")
            
            min_date_allowed = kolom_data.index.min() + pd.DateOffset(days=30)
            max_date_allowed = kolom_data.index.max()
            
            # --- PERUBAHAN: Memastikan inisialisasi session state sudah benar ---
            if 'end_date_val' not in st.session_state:
                st.session_state.end_date_val = max_date_allowed.date()
            if 'start_date_val' not in st.session_state:
                st.session_state.start_date_val = (max_date_allowed - pd.DateOffset(days=29)).date() # Gunakan 29

            def sync_dates_validation():
                if st.session_state.start_date_val > st.session_state.end_date_val:
                    st.session_state.start_date_val = st.session_state.end_date_val

            end_validation_date = st.date_input(
                "1. Pilih Tanggal Awal Prediksi",
                min_value=min_date_allowed.date(),
                max_value=max_date_allowed.date(),
                key='end_date_val',
                on_change=sync_dates_validation
            )
            
            start_validation_date = st.date_input(
                "2. Pilih Tanggal Akhir Prediksi",
                min_value=min_date_allowed.date(),
                max_value=end_validation_date,
                key='start_date_val',
                on_change=sync_dates_validation,
                help="Maksimal Prediksi 6 Bulan ke Belakang",
            )

            n_days_validation = (end_validation_date - start_validation_date).days + 1
            
            message = f"Rentang Prediksi Saat Ini: {n_days_validation} Hari"
            if n_days_validation > 180 or n_days_validation < 1:
                st.error(message)
                validate_range_button = st.button("Mulai Prediksi Mundur", type="primary", use_container_width=True, disabled=True)
            else:
                st.info(message)
                validate_range_button = st.button("Mulai Prediksi Mundur", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.info(
            "**Tentang Model:**\n"
            "Prediksi ini dibuat menggunakan model time-series LSTM (Long Short-Term Memory). "
            "Akurasi model dievaluasi menggunakan RMSE dan MAPE pada data validasi."
        )

    data_to_plot = kolom_data

    # --- Tampilan Utama ---
    st.header(f"Visualisasi dan Prediksi Harga Beras {jenis_beras}")

    with st.expander("🔍 Cek Harga Historis (Rentang Maksimal 1 Bulan)"):
        
        min_date_check = kolom_data.index.min()
        max_date_check = kolom_data.index.max()
        
        if 'expander_start' not in st.session_state:
            st.session_state.expander_start = (max_date_check - pd.DateOffset(days=7)).date()
        if 'expander_end' not in st.session_state:
            st.session_state.expander_end = max_date_check.date()
        
        def sync_expander_dates(changed_key):
            start = st.session_state.expander_start
            end = st.session_state.expander_end
            
            if (end - start).days > 30:
                st.session_state.toast_message = "Rentang disesuaikan otomatis (maksimal 1 bulan)."
                if changed_key == 'expander_start':
                    new_end = start + pd.DateOffset(days=30)
                    st.session_state.expander_end = min(new_end.date(), max_date_check.date())
                elif changed_key == 'expander_end':
                    new_start = end - pd.DateOffset(days=30)
                    st.session_state.expander_start = max(new_start.date(), min_date_check.date())
            
            if st.session_state.expander_start > st.session_state.expander_end:
                if changed_key == 'expander_start':
                    st.session_state.expander_end = st.session_state.expander_start
                else:
                    st.session_state.expander_start = st.session_state.expander_end

        col_start, col_end = st.columns(2)
        with col_start:
            start_date_check = st.date_input(
                "Tanggal Mulai",
                min_value=min_date_check.date(),
                max_value=max_date_check.date(),
                key='expander_start',
                on_change=sync_expander_dates,
                args=('expander_start',)
            )

        with col_end:
            end_date_check = st.date_input(
                "Tanggal Akhir",
                min_value=start_date_check,
                max_value=max_date_check.date(),
                key='expander_end',
                on_change=sync_expander_dates,
                args=('expander_end',)
            )
        
        if 'toast_message' in st.session_state:
            st.toast(st.session_state.toast_message, icon="⚠️")
            del st.session_state.toast_message

        if st.button("Tampilkan Data Historis", use_container_width=True):
            historical_slice = kolom_data.loc[start_date_check:end_date_check]
            if historical_slice.empty:
                st.warning("Tidak ada data yang tersedia pada rentang tanggal yang dipilih.")
            else:
                display_df = historical_slice.copy().reset_index()
                display_df['Tanggal'] = display_df['Tanggal'].dt.strftime('%Y-%m-%d')
                display_df = display_df.rename(columns={jenis_beras: "Harga (Rp)"})
                st.dataframe(
                    display_df.style.format({"Harga (Rp)": "Rp {:,.0f}"}),
                    use_container_width=True,
                    hide_index=True
                )

    fig = go.Figure()
    
    if validate_range_button:
        start_val_ts = pd.to_datetime(start_validation_date)
        end_val_ts = pd.to_datetime(end_validation_date)
        
        plot_start_date = end_val_ts - pd.DateOffset(months=3)
        plot_end_date = end_val_ts + pd.DateOffset(months=1)
        data_to_plot = kolom_data.loc[plot_start_date:plot_end_date]
        
        with st.spinner(f'Melakukan prediksi mundur untuk {n_days_validation} hari ke belakang...'):
            df_validation = pd.DataFrame()
            try:
                actual_validation_range = kolom_data.loc[start_val_ts:end_val_ts]
                if len(actual_validation_range) < 1:
                     st.warning("Tidak ada data yang ditemukan pada rentang tanggal yang dipilih.")
                else:
                    window_size = 30
                    date_list_validation, actual_prices_validation, validation_predictions = [], [], []
                    for date_to_predict in actual_validation_range.index:
                        loc_current_date = kolom_data.index.get_loc(date_to_predict)
                        if loc_current_date < window_size: continue
                        
                        start_pos = loc_current_date - window_size
                        end_pos = loc_current_date
                        validation_window = kolom_data.iloc[start_pos:end_pos]
                        
                        scaled_window = scaler.transform(validation_window.values.reshape(-1, 1))
                        input_batch = scaled_window.reshape(1, 1, window_size)
                        predicted_scaled = model.predict(input_batch, verbose=0)[0]
                        predicted_price = scaler.inverse_transform([predicted_scaled])[0][0]
                        
                        validation_predictions.append(predicted_price)
                        date_list_validation.append(date_to_predict)
                        actual_prices_validation.append(kolom_data.loc[date_to_predict][jenis_beras])
                    
                    if date_list_validation:
                        df_validation = pd.DataFrame({'Tanggal': date_list_validation, 'Harga Aktual': actual_prices_validation, 'Hasil Prediksi': validation_predictions})
                        df_validation['Selisih'] = df_validation['Hasil Prediksi'] - df_validation['Harga Aktual']
                        
                        df_validation = df_validation.sort_values(by='Tanggal', ascending=False)
                        
                        st.subheader(f"Tabel Hasil Prediksi Mundur Harga Beras {jenis_beras} ({n_days_validation} Hari ke Belakang)")
                        df_validation_display = df_validation.copy()
                        df_validation_display['Tanggal'] = df_validation_display['Tanggal'].dt.strftime('%Y-%m-%d')
                        st.dataframe(df_validation_display.style.format({"Harga Aktual": "Rp {:,.0f}", "Hasil Prediksi": "Rp {:,.0f}", "Selisih": "Rp {:+,.0f}"}), use_container_width=True, hide_index=True)
                        fig.add_trace(go.Scatter(x=df_validation['Tanggal'], y=df_validation['Hasil Prediksi'], mode='lines', name='Prediksi (Mundur)', line=dict(color='green', dash='dot', width=3)))

                        st.subheader(f"Ringkasan Prediksi")
                        prediksi_hari_terakhir = df_validation['Hasil Prediksi'].iloc[0]
                        harga_terendah_pred = df_validation['Hasil Prediksi'].min()
                        harga_tertinggi_pred = df_validation['Hasil Prediksi'].max()
                        
                        date_before_validation = df_validation['Tanggal'].iloc[-1] - pd.DateOffset(days=1)
                        price_before_validation = kolom_data.reindex([date_before_validation], method='nearest').iloc[0][jenis_beras]
                        
                        tren_validasi_text = "Naik 📈" if prediksi_hari_terakhir > price_before_validation else "Turun 📉"
                        delta_tren_validasi_value = prediksi_hari_terakhir - price_before_validation

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Prediksi Hari Awal", f"Rp {prediksi_hari_terakhir:,.0f}")
                        col2.metric("Harga Prediksi Terendah", f"Rp {harga_terendah_pred:,.0f}")
                        col3.metric("Harga Prediksi Tertinggi", f"Rp {harga_tertinggi_pred:,.0f}")
                        col4.metric("Tren Prediksi", tren_validasi_text, delta=f"{delta_tren_validasi_value:,.0f} Rupiah")
            except Exception as e:
                st.error(f"Terjadi error saat validasi: {e}")
                
            if not df_validation.empty:
                csv_validation = convert_df_to_csv(df_validation)
                st.download_button(
                    label="📥 Download Hasil Prediksi (CSV)",
                    data=csv_validation,
                    file_name=f'prediksi_mundur_beras_{jenis_beras.lower()}_{len(df_validation)}_hari.csv',
                    mime='text/csv',
                )
                st.success("Prediksi berhasil dibuat!")

    elif start_prediction:
        if rentang_waktu == '3 Bulan Terakhir': data_to_plot = kolom_data[today - pd.DateOffset(months=3):]
        elif rentang_waktu == '6 Bulan Terakhir': data_to_plot = kolom_data[today - pd.DateOffset(months=6):]
        elif rentang_waktu == '1 Tahun Terakhir': data_to_plot = kolom_data[today - pd.DateOffset(years=1):]
        else: data_to_plot = kolom_data

        with st.spinner(f'Melakukan prediksi maju untuk {n_future} hari ke depan...'):
            window_size = 30
            predictions = predict_future(model, scaler, kolom_data, window_size, n_future)
            last_date = kolom_data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_future)
            
            df_pred = pd.DataFrame({'Tanggal': future_dates, 'Harga Prediksi (Rp)': predictions.flatten()})
            
            df_pred_display = df_pred.copy()
            df_pred_display['Tanggal'] = df_pred_display['Tanggal'].dt.strftime('%Y-%m-%d')
            st.subheader(f"Tabel Hasil Prediksi Maju Harga Beras {jenis_beras} ({n_future} Hari ke Depan)")
            st.dataframe(df_pred_display.style.format({"Harga Prediksi (Rp)": "Rp {:,.0f}"}), use_container_width=True, hide_index=True)

            st.subheader("Ringkasan Hasil Prediksi")
            col1, col2, col3, col4 = st.columns(4)
            harga_besok = predictions[0][0]
            col1.metric("Prediksi Harga Besok", f"Rp {harga_besok:,.0f}")
            harga_terendah = predictions.min()
            col2.metric("Harga Terendah", f"Rp {harga_terendah:,.0f}")
            harga_tertinggi = predictions.max()
            col3.metric("Harga Tertinggi", f"Rp {harga_tertinggi:,.0f}")
            
            tren_text = "Naik 📈" if predictions[-1] > kolom_data[jenis_beras].iloc[-1] else "Turun 📉"
            delta_tren_value = predictions[-1][0] - kolom_data[jenis_beras].iloc[-1]
            col4.metric("Tren Prediksi", tren_text, delta=f"{delta_tren_value:,.0f} Rupiah")
            
            csv_pred = convert_df_to_csv(df_pred)
            st.download_button(label="📥 Download Hasil Prediksi (CSV)", data=csv_pred, file_name=f'prediksi_maju_beras{jenis_beras.lower()}_{n_future}_hari.csv', mime='text/csv')
            st.success("Prediksi berhasil dibuat!")
            fig.add_trace(go.Scatter(x=df_pred['Tanggal'], y=df_pred['Harga Prediksi (Rp)'], mode='lines', name='Hasil Prediksi Maju', line=dict(color='orange', dash='dash', width=3)))
    
    fig.add_trace(go.Scatter(x=data_to_plot.index, y=data_to_plot[jenis_beras].values, mode='lines', name='Data Historis', line=dict(color='blue', width=2)))

    fig.update_layout(title=f'Grafik Harga Beras {jenis_beras}', xaxis_title='Tanggal', yaxis_title='Harga (Rp)', legend_title='Keterangan', template='plotly_white', height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Lihat Analisis Data Historis (1 Tahun Terakhir)"):
        data_1_tahun = data[today - pd.DateOffset(years=1):]
        harga_rata_rata = data_1_tahun[jenis_beras].mean()
        harga_min_hist = data_1_tahun[jenis_beras].min()
        harga_max_hist = data_1_tahun[jenis_beras].max()
        col_hist1, col_hist2, col_hist3 = st.columns(3)
        col_hist1.metric("Harga Rata-rata (1 Thn)", f"Rp {harga_rata_rata:,.0f}")
        col_hist2.metric("Harga Terendah (1 Thn)", f"Rp {harga_min_hist:,.0f}")
        col_hist3.metric("Harga Tertinggi (1 Thn)", f"Rp {harga_max_hist:,.0f}")