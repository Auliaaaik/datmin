import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediction")
st.title("Predictions")

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Gagal Jantung", layout="centered")
st.title("🫀 Prediksi Gagal Jantung Berdasarkan Parameter Klinis")

# Fungsi load model
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Load model
model = load_model("model/random_forest_model.pkl")

# Form input
st.markdown("### Masukkan Data Pasien:")

age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=50)

sex = st.selectbox("Jenis Kelamin", options=["Laki-laki", "Perempuan"])
sex = 1 if sex == "Laki-laki" else 0

restingbp = st.number_input("Tekanan Darah Istirahat (mm Hg)", min_value=80, max_value=200, value=120)

cholesterol = st.number_input("Kadar Kolesterol (mg/dL)", min_value=100, max_value=600, value=240)

fastingbs = st.selectbox("Gula Darah Puasa > 120 mg/dL?", options=["Tidak", "Ya"])
fastingbs = 1 if fastingbs == "Ya" else 0

restingecg = st.selectbox("Hasil EKG saat istirahat", options=[
    "Normal", "Memiliki kelainan gelombang ST-T", "Menunjukkan hipertrofi ventrikel kiri"
])
# Asumsikan encode:
# 0: Normal, 1: ST-T abnormality, 2: LV hypertrophy
restingecg_mapping = {
    "Normal": 0,
    "Memiliki kelainan gelombang ST-T": 1,
    "Menunjukkan hipertrofi ventrikel kiri": 2
}
restingecg = restingecg_mapping[restingecg]

maxhr = st.number_input("Denyut Jantung Maksimum", min_value=60, max_value=220, value=150)

exerciseangina = st.selectbox("Apakah mengalami angina karena olahraga?", options=["Tidak", "Ya"])
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

# Prediksi
if st.button("Prediksi Gagal Jantung"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # Probabilitas positif

    if prediction == 1:
        st.error(f"⚠️ Pasien kemungkinan mengalami gagal jantung (Probabilitas: {probability:.2%})")
    else:
        st.success(f"✅ Pasien **tidak** berisiko gagal jantung (Probabi
