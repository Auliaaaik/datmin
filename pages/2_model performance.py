import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Konfigurasi halaman
st.set_page_config(page_title="Model Prediksi: Heart Disease", layout="centered")
st.title("🫀 Model Prediksi: Heart Disease")

# Load dataset
df = pd.read_csv("model/Gagal_Jantung.csv", sep=';')

# Pilih ukuran data testing
test_size_percent = st.slider("Data Testing (%)", 10, 90, 20)
st.write(f"Nilai yang dipilih: {test_size_percent}%")

# Tombol trigger
if st.button("Hasil"):

    # Bagi data
    test_size = test_size_percent / 100
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Latih model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    st.subheader("📊 Ringkasan Metrik")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi", f"{acc:.2%}")
    col2.metric("Presisi", f"{prec:.2%}")
    col3.metric("Recall", f"{rec:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}")

    # Confusion Matrix
    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Tidak", "Ya"], yticklabels=["Tidak", "Ya"])
    ax_cm.set_xlabel("Prediksi")
    ax_cm.set_ylabel("Aktual")
    st.pyplot(fig_cm)

    # Classification Report
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="YlOrBr"), use_container_width=True)

    # Visualisasi metrik
    st.subheader("📌 Visualisasi Skor Model")
    metrics = [acc, prec, rec, f1]
    names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    fig, ax = plt.subplots()
    sns.barplot(x=names, y=metrics, palette="Set2", ax=ax)
    ax.set_ylim(0, 1)
    for i, v in enumerate(metrics):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("Model ini menggunakan **Logistic Regression** untuk memprediksi kemungkinan gagal jantung berdasarkan data pasien.")
