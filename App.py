import streamlit as st
from PIL import Image

# Load gambar dari folder assets
image = Image.open("assets/jantung.png")

# Tampilkan gambar
st.image(image, caption="", use_container_width=True)

# Judul Aplikasi
st.title("Selamat Datang di Aplikasi Prediksi Penyakit Gagal Jantung")

# Deskripsi Penjelasan
st.markdown("""
Aplikasi ini dirancang untuk membantu pengguna dalam **memprediksi risiko gagal jantung**
berdasarkan berbagai **gejala klinis** dan **parameter kesehatan** pasien.

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
