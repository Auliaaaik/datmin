import streamlit as st
import pandas as pd
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Prediction", layout="centered")
st.title("🫀 Prediksi Gagal Jantung Berdasarkan Parameter Klinis")

# 🔍 Deskripsi Singkat
st.write("""
Aplikasi ini digunakan untuk memprediksi kemungkinan seseorang mengalami **gagal jantung**
berdasarkan **parameter klinis** seperti tekanan darah, kadar kolesterol, hasil EKG, dan lainnya.  
Model yang digunakan adalah **Random Forest Classifier** yang dilatih dari data riwayat medis pasien.
""")

# Fungsi load model
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Load model
model = load_model("model/random_forest_model.pkl")

# --- Input Formulir dengan keterangan ---
age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=50)
st.caption("🧓 Umur pasien dalam tahun.")

sex = st.selectbox("Jenis Kelamin", options=["Laki-laki", "Perempuan"])
st.caption("🚻 Pilih jenis kelamin pasien.")
sex = 1 if sex == "Laki-laki" else 0

restingbp = st.number_input("Tekanan Darah Istirahat (mm Hg)", min_value=80, max_value=200, value=120)
st.caption("🩺 Tekanan darah saat pasien dalam kondisi istirahat.")

cholesterol = st.number_input("Kadar Kolesterol (mg/dL)", min_value=100, max_value=600, value=240)
st.caption("🍳 Jumlah kolesterol total dalam darah.")

fastingbs = st.selectbox("Gula Darah Puasa > 120 mg/dL?", options=["Tidak", "Ya"])
st.caption("🧪 Apakah kadar gula darah puasa pasien lebih dari 120 mg/dL?")
fastingbs = 1 if fastingbs == "Ya" else 0

restingecg = st.selectbox("Hasil EKG saat istirahat", options=[
    "Normal", "Memiliki kelainan gelombang ST-T", "Menunjukkan hipertrofi ventrikel kiri"
])
st.caption("📈 Hasil elektrokardiogram (EKG) saat pasien beristirahat.")
restingecg_mapping = {
    "Normal": 0,
    "Memiliki kelainan gelombang ST-T": 1,
    "Menunjukkan hipertrofi ventrikel kiri": 2
}
restingecg = restingecg_mapping[restingecg]

maxhr = st.number_input("Denyut Jantung Maksimum", min_value=60, max_value=220, value=150)
st.caption("❤️‍🔥 Detak jantung maksimum yang dicapai pasien saat berolahraga.")

exerciseangina = st.selectbox("Apakah mengalami angina karena olahraga?", options=["Tidak", "Ya"])
st.caption("🏃‍♂️ Apakah pasien mengalami nyeri dada (angina) saat aktivitas fisik?")
exerciseangina = 1 if exerciseangina == "Ya" else 0

# Buat DataFrame untuk prediksi
input_df = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'RestingBP': [restingbp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fastingbs],
    'RestingECG': [restingecg],
    'MaxHR': [maxhr],
    'ExerciseAngina': [exerciseangina]
})

# Tombol Prediksi
if st.button("🔍 Prediksi Gagal Jantung"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # Probabilitas positif

    if prediction == 1:
        st.error(f"⚠️ Berdasarkan data yang dimasukkan, pasien kemungkinan mengalami **gagal jantung**.\n\n**Probabilitas: {probability:.2%}**")
    else:
        st.success(f"✅ Berdasarkan data yang dimasukkan, pasien **tidak berisiko mengalami gagal jantung**.\n\n**Probabilitas: {probability:.2%}**")
        
        st.caption("Hasil prediksi berdasarkan atribut atribut yang dimasukkan, hasil tidak selalu tepat, selalu konsultasikan ke dokter untuk hasil yang lebih akurat") 
