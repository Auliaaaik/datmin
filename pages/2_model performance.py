import streamlit as st

st.set_page_config(page_title="Model Performance")
st.title("Model Performance")

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Judul dan konfigurasi
st.set_page_config(page_title="Model Akurasi")
st.title("Akurasi Model Klasifikasi")

# Baca dataset
df = pd.read_csv("model/Gagal_jantung.csv", sep=';')

# Tampilkan dataset
st.subheader("Isi Dataset:")
st.dataframe(df)

# Pisahkan fitur dan target (asumsikan kolom target bernama 'DEATH_EVENT')
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Split data ke training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Melatih model
model = LogisticRegression(max_iter=1000)  # max_iter dinaikkan agar konvergen
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)

# Tampilkan akurasi
st.subheader("Akurasi Model Logistic Regression:")
st.write(f"{accuracy:.2%}")
