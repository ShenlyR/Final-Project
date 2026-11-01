import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load Model, Fitur, dan Freq Maps
model = xgb.XGBRegressor()
model.load_model('xgboost_model.json')
features = joblib.load('model_features.joblib')
freq_maps = joblib.load('freq_maps.joblib')

# Konfigurasi Halaman 
st.set_page_config(page_title="Dashboard Pendapatan Toko", layout="wide")
st.title("📊 Dashboard Pendapatan Bersih Toko")
st.write("Prediksi pendapatan bersih Anda berdasarkan pesanan yang masuk")

# Buat Form Input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Pesanan")
    total_qty = st.number_input("Total Kuantitas Barang", min_value=1, value=1)
    total_weight = st.number_input("Total Berat Paket (gram)", min_value=10, value=500)
    num_categories = st.number_input("Jumlah Kategori Produk", min_value=1, value=1)
    
    product_category = st.selectbox("Kategori Produk Utama", list(freq_maps['product_categories'].keys()))
    shipping_option = st.selectbox("Opsi Pengiriman", list(freq_maps['Opsi Pengiriman'].keys()))

with col2:
    st.subheader("Biaya & Status")
    payment_method = st.selectbox("Metode Pembayaran", list(freq_maps['Metode Pembayaran'].keys()))
    city = st.selectbox("Kota/Kabupaten Tujuan", list(freq_maps['Kota/Kabupaten'].keys()))
    
    ongkir_dibayar = st.number_input("Simulasi Ongkos Kirim Dibayar Pembeli (Rp)", min_value=0, value=0)
    diskon = st.number_input("Simulasi Total Diskon Diberikan (Rp)", min_value=0, value=0)
    
    status_pesanan = st.selectbox("Status Pesanan", ["Selesai (Diterima)", "Batal"])

# --- PERUBAHAN: Tombol dikecilkan dengan kolom ---
st.write("") # Spasi kosong sedikit
col_btn, _ = st.columns([1, 5]) 
with col_btn:
    tombol_prediksi = st.button("Total Pendapatan", type="primary", use_container_width=True)

# Proses Prediksi
if tombol_prediksi:
    # Transformasi Log
    total_qty_log = np.log1p(total_qty)
    total_weight_log = np.log1p(total_weight)
    diskon_log = np.log1p(diskon)
    ongkir_log = np.log1p(ongkir_dibayar)

    # Encode kategori
    product_freq = freq_maps['product_categories'].get(product_category, 0)
    shipping_freq = freq_maps['Opsi Pengiriman'].get(shipping_option, 0)
    payment_freq = freq_maps['Metode Pembayaran'].get(payment_method, 0)
    city_freq = freq_maps['Kota/Kabupaten'].get(city, 0)

    # Susun DataFrame
    input_data = {
        'num_product_categories': num_categories,
        'total_qty_log': total_qty_log,
        'total_weight_gr_log': total_weight_log,  
        'Total Diskon_log': diskon_log,
        'Ongkos Kirim Dibayar oleh Pembeli_log': ongkir_log,
        'product_categories_freq': product_freq,
        'Opsi Pengiriman_freq': shipping_freq,
        'Metode Pembayaran_freq': payment_freq,
        'Kota/Kabupaten_freq': city_freq
    }
    
    input_df = pd.DataFrame([input_data])[features]

    # Logika Prediksi
    if status_pesanan == "Batal":
        prediction_rupiah = 0.0
    else:
        prediction_log = model.predict(input_df)
        prediction_rupiah = np.expm1(prediction_log)[0]

    # --- HASIL PREDIKSI ---
    st.divider()
    
    # Card Hasil Utama
    with st.container(border=True):
        st.subheader("💳 Hasil Transaksi")
        if status_pesanan == "Batal":
            st.error(f"**Status: Pesanan Dibatalkan**")
            st.warning(f"Potensi Pendapatan Bersih: **Rp 0** (Transaksi gagal/hangus).")
        else:
            st.success(f"**Status: Pesanan Selesai/Diterima**")
            st.metric(label="Estimasi Pendapatan Bersih", value=f"Rp {prediction_rupiah:,.2f}")

    # --- RINCIAN FINANSIAL ---
    st.markdown("### 📊 Rincian Finansial Transaksi")
    with st.container(border=True):
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric(label="Total Diskon Diberikan", value=f"Rp {diskon:,.0f}")
        with col_res2:
            st.metric(label="Ongkir Ditanggung Pembeli", value=f"Rp {ongkir_dibayar:,.0f}")
        with col_res3:
            estimasi_kotor = prediction_rupiah + diskon
            st.metric(label="Estimasi Nilai Kotor (Sebelum Diskon)", value=f"Rp {estimasi_kotor:,.0f}")

    # --- GRAFIK ALUR PENDAPATAN (WATERFALL) ---
    st.markdown("### 📈 Grafik Alur Pendapatan")
    
    with st.container(border=True):
        if status_pesanan == "Batal" or prediction_rupiah == 0:
            st.info("Grafik tidak tersedia karena tidak ada pendapatan (pesanan batal).")
        else:
            harga_kotor = prediction_rupiah + diskon
            
            labels = ['Harga Kotor', 'Diskon', 'Pendapatan Bersih']
            values = [harga_kotor, -diskon, prediction_rupiah]
            colors = ['#2ca02c', '#d62728', '#1f77b4'] 
            
            fig, ax = plt.subplots(figsize=(10, 4))
            
            ax.bar(labels[0], values[0], color=colors[0])
            
            if diskon > 0:
                ax.bar(labels[1], values[1], bottom=prediction_rupiah, color=colors[1])
            
            ax.bar(labels[2], values[2], color=colors[2])
            
            for i, (label, val) in enumerate(zip(labels, values)):
                if i == 0:
                    y_pos = val
                elif i == 1:
                    y_pos = prediction_rupiah + abs(val) if diskon > 0 else prediction_rupiah
                else:
                    y_pos = val
                    
                if val != 0:
                    text_val = abs(val)
                    ax.text(i, y_pos + (max(values)*0.02), f'Rp {text_val:,.0f}', 
                            ha='center', va='bottom', fontweight='bold', color='black')
                
            ax.set_ylabel('Nominal (Rp)')   
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'Rp {x:,.0f}'))
            ax.set_ylim(0, max(values) * 1.2)
            
            ax.plot([0.4, 0.6], [harga_kotor, harga_kotor], color='gray', linestyle='--', linewidth=1)
            ax.plot([1.4, 1.6], [prediction_rupiah, prediction_rupiah], color='gray', linestyle='--', linewidth=1)
            
            st.pyplot(fig)