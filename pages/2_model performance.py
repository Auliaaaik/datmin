import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Page config
st.set_page_config(page_title="Model Performance", layout="wide")
st.title("📈 Evaluasi Model Prediksi Gagal Jantung")

# Load data
df = pd.read_csv("model/Gagal_Jantung.csv", sep=';')

# Preprocessing
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Menentukan kelas positif
unique_labels = sorted(set(y_test) | set(y_pred))
if len(unique_labels) < 2:
    # hanya 1 kelas yang muncul
    acc = accuracy_score(y_test, y_pred)
    st.subheader("📊 Ringkasan Metrik")
    st.metric("Akurasi", f"{acc:.2%}")
    st.warning("⚠️ Model tidak bisa menghitung precision, recall, dan F1-score karena hanya satu kelas yang muncul.")
else:
    pos_label = 1 if 1 in unique_labels else unique_labels[0]

    # Hitung metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)

    # Tampilkan metrik
    st.subheader("📊 Ringkasan Metrik")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi", f"{acc:.2%}")
    col2.metric("Presisi", f"{prec:.2%}")
    col3.metric("Recall", f"{rec:.2%}")
    col4.metric("F1-Score", f"{f1:.2%}")

    # Confusion Matrix
    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    ax_cm.set_xlabel("Prediksi")
    ax_cm.set_ylabel("Aktual")
    st.pyplot(fig_cm)

    # Classification Report
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Oranges'), use_container_width=True)

    # Visualisasi skor metrik
    st.subheader("📌 Visualisasi Skor Model")
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_values = [acc, prec, rec, f1]
    fig_score, ax_score = plt.subplots()
    sns.barplot(x=metric_values, y=metric_names, palette='Set2', ax=ax_score)
    ax_score.set_xlim(0, 1)
    ax_score.set_title("Skor Evaluasi Model", fontsize=14)
    st.pyplot(fig_score)

# Footer
st.markdown("---")
st.markdown("Model menggunakan **Logistic Regression** untuk prediksi penyakit jantung berdasarkan atribut pasien.")
