import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Model Performance")
st.title("Model Performance")

# Konfigurasi halaman
st.set_page_config(page_title="Akurasi Model Random Forest")
st.title("Akurasi Model: Random Forest")

# Load dataset
df = pd.read_csv("model/Gagal_jantung.csv", sep=';')
