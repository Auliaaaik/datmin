import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Penyakit Jantung")
st.title("Model Prediksi: Heart Disease")

# Load dataset
df = pd.read_csv("model/Gagal_Jantung.csv", sep=';')

testing= st.slider ("Data Testing", min_value=10, max_value=90, value=20)
st.writer(f"Nilai yang dipilih: {testing})
t_size = testing/100

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def load_model(path):
    model = joblib.load(path)
    return model

model = load_model('model/random_forest_model.pkl')

if st.button("Hasil"):
          
