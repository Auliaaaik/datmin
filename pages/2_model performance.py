import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="wide")
st.title("🫀 Model Prediksi: Heart Disease")

# 🔍 Deskripsi Singkat
st.markdown("""
Model ini bertujuan untuk memprediksi apakah seorang pasien mengalami **penyakit jantung** atau tidak, 
berdasarkan data historis pasien yang telah dilabeli. Model yang digunakan adalah **Random Forest Classifier**, 
yang terbukti efektif dalam klasifikasi berbasis data medis.
""")

# Load dataset
df = pd.read_csv("model/Gagal_Jantung.csv", sep=';')

# UI: Slider untuk memilih data testing
st.sidebar.header("🔧 Pengaturan")
testing = st.sidebar.slider("Persentase Data Testing", min_value=10, max_value=90, value=20)
t_size = testing / 100

st.sidebar.markdown(f"📊 **Data testing yang dipilih:** {testing}%")

# Split data
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=42)

# Load model
@st.cache_resource
def load_model(path):
    model = joblib.load(path)
    return model

model = load_model('model/random_forest_model.pkl')

# Tombol prediksi
if st.button("🔍 Tampilkan Hasil"):
    # Prediksi dengan model
    y_pred = model.predict(X_test)

    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("✅ Akurasi Model")
    st.metric(label="Akurasi Random Forest", value=f"{accuracy:.2%}")

    # Classification Report
    st.subheader("📄 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()

    # Tampilan tabel yang lebih interaktif
    st.dataframe(report_df.style.background_gradient(cmap="YlGnBu"), use_container_width=True)

    # Plot bar skor metrik utama
    st.subheader("📊 Visualisasi Skor Metrik")
    scores = report_df.loc[['accuracy', 'precision', 'recall', 'f1-score']].dropna()
    fig, ax = plt.subplots()
    sns.barplot(x=scores.index, y=scores['support'], palette='Set2')
    ax.set_title("Support Tiap Metrik", fontsize=12)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("🧠 Model menggunakan algoritma **Random Forest Classifier** untuk memprediksi kemungkinan penyakit jantung berdasarkan data pasien.")
