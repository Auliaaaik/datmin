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

