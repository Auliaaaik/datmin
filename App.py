import streamlit as st

# Konfigurasi layout biar lebar penuh
st.set_page_config(layout="wide")

# Tambahkan gambar ilustrasi jantung
st.image(
    "https://cdn.pixabay.com/photo/2016/03/31/19/14/heart-1297622_960_720.png",
    use_container_width=True  # Ini pengganti use_column_width
)

# Judul Aplikasi
st.title("Selamat Datang di Aplikasi Prediksi Penyakit Gagal Jantung")

# Deskripsi Penjelasan
st.markdown("""
Aplikasi ini dirancang untuk membantu pengguna dalam **memprediksi risiko gagal jantung**
berdasarkan berbagai **gejala klinis** dan **parameter kesehatan** yang dimasukkan.

---

### 💡 Apa yang bisa dilakukan aplikasi ini?
- ✅ Menginput data kesehatan seperti tekanan darah, kadar kolesterol, detak jantung, dll.
- ✅ Memproses data dengan model Machine Learning.
- ✅ Memberikan hasil prediksi risiko **tinggi** atau **rendah** terhadap penyakit gagal jantung.

---

### 🎯 Tujuan Aplikasi:
- Membantu masyarakat melakukan deteksi dini terhadap gagal jantung.
- Menyediakan alat bantu diagnosis untuk edukasi dan konsultasi awal.

---

> ⚠️ **Catatan Penting:**  
> Prediksi dari aplikasi ini **tidak menggantikan diagnosis dokter**. Silakan konsultasikan hasil prediksi ke profesional medis.
""")
