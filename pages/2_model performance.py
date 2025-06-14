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

# Tampilkan dataset
st.subheader("Isi Dataset:")
st.dataframe(df)

# Cek apakah kolom target ada
if 'DEATH_EVENT' not in df.columns:
    st.error("Kolom 'DEATH_EVENT' tidak ditemukan dalam dataset!")
else:
    # Pisahkan fitur dan target
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Buat model Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi dan akurasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Tampilkan hasil
    st.subheader("Akurasi Model Random Forest:")
    st.success(f"Akurasi: {accuracy:.2%}")

