import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns 

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

# Button
st.write("") 
col_btn, _ = st.columns([1, 3]) 
with col_btn:
    tombol_prediksi = st.button("Simulasi Pendapatan", type="primary", use_container_width=True)

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

    # Hasil
    st.divider()
    
    # Hasil Utama (Tanpa Card)
    st.subheader("💳 Hasil Transaksi")
    if status_pesanan == "Batal":
        st.error(f"**Status: Pesanan Dibatalkan**")
        st.warning(f"Potensi Pendapatan Bersih: **Rp 0** (Transaksi gagal/hangus).")
    else:
        st.success(f"**Status: Pesanan Selesai/Diterima**")
        st.metric(label="Estimasi Pendapatan Bersih", value=f"Rp {prediction_rupiah:,.2f}")

    # Informasi (Tanpa Card)
    st.markdown("### 📊 Rincian Finansial Transaksi")
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.metric(label="Total Diskon Diberikan", value=f"Rp {diskon:,.0f}")
    with col_res2:
        st.metric(label="Ongkir Ditanggung Pembeli", value=f"Rp {ongkir_dibayar:,.0f}")
    with col_res3:
        estimasi_kotor = prediction_rupiah + diskon
        st.metric(label="Estimasi Nilai Kotor (Sebelum Diskon)", value=f"Rp {estimasi_kotor:,.0f}")

    # Visualisasi Grafik 
    st.markdown("Grafik Pendapatan")
    
    if status_pesanan == "Batal" or prediction_rupiah == 0:
        st.info("Grafik tidak tersedia karena tidak ada pendapatan (pesanan batal).")
    else:
        harga_kotor = prediction_rupiah + diskon
        
        # Menggunakan nilai positif 
        labels = ['Harga Kotor', 'Diskon', 'Pendapatan Bersih']
        values = [harga_kotor, diskon, prediction_rupiah]

        palette = sns.color_palette("viridis", 3) 
        colors = [palette[0], palette[1], palette[2]]
        
        col_graph, col_empty = st.columns([1, 1]) 
        
        with col_graph:
            fig, ax = plt.subplots(figsize=(7, 4)) 
            bars = ax.bar(labels, values, color=colors)
            
            # Menambahkan teks nominal 
            for bar in bars:
                yval = bar.get_height()
                if yval > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), 
                            f'Rp {yval:,.0f}', 
                            ha='center', va='bottom', fontweight='bold', color='black', fontsize=9)
                
            ax.set_ylabel('Nominal (Rp)')        
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'Rp {x:,.0f}'))
            ax.set_ylim(0, max(values) * 1.2)
            st.pyplot(fig, use_container_width=True)