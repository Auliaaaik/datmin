import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard - Gagal Jantung", layout="wide")

# Judul utama
st.title("🫀 Dashboard Prediksi Gagal Jantung")
st.markdown(
    """
    Aplikasi ini bertujuan untuk membantu memprediksi kemungkinan **gagal jantung** berdasarkan berbagai parameter medis.
    Dataset yang digunakan merupakan kumpulan data pasien dengan berbagai fitur seperti tekanan darah, kolesterol, dan detak jantung maksimal.
    """
)

# Membaca dataset
df = pd.read_csv("model/Gagal_Jantung.csv", sep=';')

# Info ukuran dataset
st.markdown(f"📊 **Jumlah Data:** {df.shape[0]} baris, {df.shape[1]} kolom")

# Tampilkan isi dataset
st.subheader("📄 Isi Dataset")
st.dataframe(df, use_container_width=True)

# Statistik deskriptif
st.subheader("📈 Statistik Ringkas")
st.write("Statistik umum dari fitur-fitur numerik yang ada di dataset.")
st.dataframe(df.describe(), use_container_width=True)

# Visualisasi tambahan: Distribusi umur & kolesterol
st.subheader("🔍 Visualisasi Data")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Distribusi Umur**")
    fig1, ax1 = plt.subplots()
    df["Age"].hist(bins=20, color="#1f77b4", edgecolor="black", ax=ax1)
    ax1.set_xlabel("Usia")
    ax1.set_ylabel("Jumlah Pasien")
    st.pyplot(fig1)

with col2:
    st.markdown("**Distribusi Kolesterol**")
    fig2, ax2 = plt.subplots()
    df["Cholesterol"].hist(bins=20, color="#ff7f0e", edgecolor="black", ax=ax2)
    ax2.set_xlabel("Kolesterol")
    ax2.set_ylabel("Jumlah Pasien")
    st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("🧠 *Dashboard ini dibuat sebagai bagian dari proyek Data Mining Gagal Jantung.*")
