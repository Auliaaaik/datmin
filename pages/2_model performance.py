import streamlit as st
import pandas as pd

# Konfigurasi halaman (hanya sekali, di awal)
st.set_page_config(page_title="Akurasi Model Random Forest")
st.title("Akurasi Model: Random Forest")

# Load dataset
df = pd.read_csv("model/Gagal_jantung.csv", sep=';')

# Tampilkan dataset
st.subheader("Dataset Gagal Jantung:")
st.dataframe(df)

# Cek apakah kolom target tersedia
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

    # Buat dan latih model Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi dan evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Tampilkan hasil
    st.subheader("Akurasi Model:")
    st.success(f"Akurasi Random Forest: {accuracy:.2%}")
