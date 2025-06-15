import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(page_title="Heart Failure Dashboard", layout="wide")
st.title("📊 Heart Failure Dataset Dashboard")

# Deskripsi Dataset
st.markdown("""
Selamat datang di dashboard analisis dataset **Gagal Jantung**.  
Dataset ini berisi informasi medis dari pasien untuk menganalisis risiko gagal jantung.
""")

# Membaca dataset
df = pd.read_csv("model/Gagal_Jantung.csv", sep=';')

# Tampilkan isi dataset
st.subheader("📂 Isi Dataset")
st.dataframe(df, use_container_width=True)

# Tampilkan info statistik
st.subheader("📈 Statistik Umum")
col1, col2, col3 = st.columns(3)
col1.metric("🧓 Rata-rata Usia", f"{df['Age'].mean():.1f} tahun")
col2.metric("🩸 Rata-rata Tekanan Darah", f"{df['RestingBP'].mean():.1f}")
col3.metric("💉 Rata-rata Kolesterol", f"{df['Cholesterol'].mean():.1f}")

st.write(df.describe())

# Visualisasi: Distribusi umur
st.subheader("📊 Distribusi Usia Pasien")
fig1, ax1 = plt.subplots()
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue', ax=ax1)
ax1.set_xlabel("Usia")
ax1.set_ylabel("Jumlah Pasien")
st.pyplot(fig1)

# Visualisasi: Korelasi antar fitur
st.subheader("🧠 Korelasi Antar Fitur")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
st.pyplot(fig2)

# Visualisasi: Distribusi Jenis Kelamin
st.subheader("👨‍⚕️👩‍⚕️ Distribusi Jenis Kelamin")
gender_counts = df['Sex'].value_counts()
fig3, ax3 = plt.subplots()
ax3.pie(gender_counts, labels=["Pria", "Wanita"], autopct='%1.1f%%', colors=['#4C72B0','#55A868'], startangle=90)
ax3.axis("equal")
st.pyplot(fig3)

# Footer
st.markdown("---")
st.markdown("Dibuat dengan ❤️ oleh Tim Analisis Data")
