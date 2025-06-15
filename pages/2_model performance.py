import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Penyakit Jantung")
st.title("Model Prediksi: Heart Disease")

# 🔍 Deskripsi Singkat
st.write("""
Model ini bertujuan untuk memprediksi apakah seorang pasien mengalami **penyakit jantung** atau tidak, 
berdasarkan data historis pasien yang telah dilabeli. Model yang digunakan adalah **Random Forest Classifier**, 
yang terbukti efektif dalam klasifikasi berbasis data medis.
""")

# Load dataset
df = pd.read_csv("model/Gagal_Jantung.csv", sep=';')

testing = st.slider("Data Testing (%)", min_value=10, max_value= 90, value=20)
st.write(f"Nilai yang dipilih: {testing}%")
t_size = testing / 100

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = t_size, random_state=42)

@st.cache_resource
def load_model(path):
    model = joblib.load(path)
    return model

model = load_model('model/random_forest_model.pkl')

if st.button("Hasil"):
    # Prediksi dengan model
    y_pred = model.predict(X_test)

    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Akurasi Model:")
    st.success(f"Akurasi Random Forest: {accuracy:.2%}")

    # Classification report
    st.subheader("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    # Footer
    st.write("---")
    st.write("🧠 Model ini menggunakan **Random Forest Classifier** untuk memprediksi kemungkinan penyakit jantung berdasarkan atribut pasien.")


