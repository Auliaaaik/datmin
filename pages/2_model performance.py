import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Konfigurasi halaman
st.set_page_config(page_title="Akurasi Model Random Forest")
st.title("Akurasi Model: Random Forest")

# Load dataset
df = pd.read_csv("model/Gagal_Jantung.csv", sep=';')

# Tampilkan dataset
st.subheader("Dataset Gagal Jantung:")
st.dataframe(df)

# Cek apakah kolom target tersedia
if 'DEATH_EVENT' not in df.columns:
    st.error("Kolom 'Gagal Jantung' tidak ditemukan dalam dataset!")
else:
    # Pisahkan fitur dan target
    X = df.drop('Gagal Jantung', axis=1)
    y = df['Gagal Jantung']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Buat dan latih model Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi dan evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Tampilkan hasil akurasi
    st.subheader("Akurasi Model:")
    st.success(f"Akurasi Random Forest: {accuracy:.2%}")

    # Tampilkan classification report
    st.subheader("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
