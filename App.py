import streamlit as st

# Set halaman jadi wide
st.set_page_config(layout="wide")

# Tambahin gambar dari URL (bisa ganti kalau mau pake gambar sendiri)
st.image("https://cdn.pixabay.com/photo/2016/03/31/19/14/heart-1297622_960_720.png", use_column_width=True)

# Judul aplikasi
st.title("Selamat Datang di Aplikasi Prediksi Penyakit Gagal Jantung")

# Deskripsi tambahan
st.markdown("""
Aplikasi ini dirancang untuk membantu pengguna dalam **memprediksi risiko gagal jantung** berdasarkan berbagai **gejala klinis** dan **parameter kesehatan** yang dimasukkan.

💡 **Apa yang bisa dilakukan aplikasi ini?**
- Menginput data kesehatan seperti tekanan darah, kadar kolesterol, detak jantung, dll.
- Memproses data dengan model Machine Learning.
- Memberikan hasil prediksi risiko **tinggi** atau **rendah** terhadap penyakit gagal jantung.

🎯 **Tujuan Aplikasi:**
- Membantu masyarakat melakukan deteksi dini.
- Menyediakan alat bantu diagnosis untuk edukasi dan konsultasi awal.

🛡️ *Catatan:*
Prediksi dari aplikasi ini **tidak menggantikan diagnosis medis profesional**. Selalu konsultasikan ke dokter untuk keputusan medis lebih lanjut.

""")
