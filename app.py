import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# 1. Load Model, Fitur, dan Freq Maps
model = xgb.XGBRegressor()
model.load_model('xgboost_model.json')
features = joblib.load('model_features.joblib')
freq_maps = joblib.load('freq_maps.joblib')

# 2. Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Pendapatan Bersih", layout="wide")
st.title("📦 Prediksi Pendapatan Bersih E-Commerce")
st.write("Masukkan detail pesanan di bawah ini untuk memprediksi pendapatan bersih.")

# 3. Buat Form Input (Layout 2 Kolom)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Detail Pesanan")
    total_qty = st.number_input("Total Kuantitas Barang", min_value=1, value=1)
    total_weight = st.number_input("Total Berat (gram)", min_value=10, value=500)
    num_categories = st.number_input("Jumlah Kategori Produk", min_value=1, value=1)
    
    # Pilihan Kategori
    product_category = st.selectbox("Kategori Produk Utama", list(freq_maps['product_categories'].keys()))
    shipping_option = st.selectbox("Opsi Pengiriman", list(freq_maps['Opsi Pengiriman'].keys()))

with col2:
    st.subheader("Detail Pengiriman & Pembayaran")
    payment_method = st.selectbox("Metode Pembayaran", list(freq_maps['Metode Pembayaran'].keys()))
    city = st.selectbox("Kota/Kabupaten", list(freq_maps['Kota/Kabupaten'].keys()))
    
    ongkir_dibayar = st.number_input("Ongkos Kirim Dibayar Pembeli (Rp)", min_value=0, value=0)
    diskon = st.number_input("Total Diskon (Rp)", min_value=0, value=0)

# 4. Proses Prediksi
if st.button("Prediksi Pendapatan", type="primary"):
    # Transformasi Log (karena model dilatih dengan log1p)
    # Gunakan np.log1p
    total_qty_log = np.log1p(total_qty)
    total_weight_log = np.log1p(total_weight)
    diskon_log = np.log1p(diskon)
    ongkir_log = np.log1p(ongkir_dibayar)

    # Encode kategori dengan freq_maps
    # Jika kategori tidak ditemukan (kasus langka), isi 0
    product_freq = freq_maps['product_categories'].get(product_category, 0)
    shipping_freq = freq_maps['Opsi Pengiriman'].get(shipping_option, 0)
    payment_freq = freq_maps['Metode Pembayaran'].get(payment_method, 0)
    city_freq = freq_maps['Kota/Kabupaten'].get(city, 0)

    # Susun DataFrame sesuai urutan 'features'
    input_data = {
        'num_product_categories': num_categories,
        'total_qty_log': total_qty_log,
        'total_weight_gr_log': total_weight_log,  # <-- Diperbaiki (ditambah _gr)
        'Total Diskon_log': diskon_log,
        'Ongkos Kirim Dibayar oleh Pembeli_log': ongkir_log,
        'product_categories_freq': product_freq,
        'Opsi Pengiriman_freq': shipping_freq,
        'Metode Pembayaran_freq': payment_freq,
        'Kota/Kabupaten_freq': city_freq
    }
    
    # Pastikan urutan kolom sama persis dengan 'features'
    input_df = pd.DataFrame([input_data])[features]

    # Prediksi (Gunakan input_df, bukan input_data)
    prediction_log = model.predict(input_df) # <-- Diperbaiki
    prediction_rupiah = np.expm1(prediction_log)[0] # Kembalikan ke skala asli

    # Tampilkan Hasil
    st.success(f"💰 Estimasi Pendapatan Bersih: **Rp {prediction_rupiah:,.2f}**")