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

# Sidebar: Pengaturan
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
    y_pred = model.predict(X_test)

    # Akurasi
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("✅ Akurasi Model")
    st.metric(label="Akurasi Random Forest", value=f"{accuracy:.2%}")

    # Classification Report
    st.subheader("📄 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="YlGnBu"), use_container_width=True)

    # Visualisasi barplot
    st.subheader("📊 Visualisasi Skor Metrik")
    metric_names = ['precision', 'recall', 'f1-score']
    scores = report_df.loc[metric_names].reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='index', y='f1-score', data=scores, palette='Set2')
    ax.set_ylabel("Nilai Skor")
    ax.set_xlabel("Metrik")
    ax.set_title("Perbandingan Skor Metrik")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("🧠 Model ini menggunakan **Random Forest Classifier** untuk memprediksi kemungkinan penyakit jantung berdasarkan atribut pasien.")
