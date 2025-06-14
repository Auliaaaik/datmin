import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Penyakit Jantung")
st.title("Model Prediksi: Heart Disease")

# Load dataset
try:
    df = pd.read_csv("model/Gagal_Jantung.csv", sep=';')
    st.subheader("Dataset:")
    st.dataframe(df)
except FileNotFoundError:
    st.error("File CSV tidak ditemukan. Pastikan path dan nama file sudah benar.")

# Lanjut jika file berhasil dimuat
if 'HeartDisease' in df.columns:
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # Bagi data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi dan evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Akurasi Model:")
    st.success(f"Akurasi Random Forest: {accuracy:.2%}")

    st.subheader("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
else:
    st.warning("Kolom 'heartdisease' tidak ditemukan dalam dataset.")
