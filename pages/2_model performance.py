import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

# Konfigurasi halaman
st.set_page_config(page_title="Model Performance", layout="wide")
st.title("📈 Evaluasi Model Prediksi Gagal Jantung")

# Load dataset
df = pd.read_csv("model/Gagal_Jantung.csv", sep=';')

# Preprocessing
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Hitung metrik
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Layout metrik utama
st.subheader("📊 Ringkasan Metrik Model")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Akurasi", f"{acc:.2%}", delta_color="off")
col2.metric("Presisi", f"{prec:.2%}", delta_color="off")
col3.metric("Recall", f"{rec:.2%}", delta_color="off")
col4.metric("F1 Score", f"{f1:.2%}", delta_color="off")

# Visualisasi skor metrik
st.markdown("#### 📌 Visualisasi Kinerja")
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_values = [acc, prec, rec, f1]
fig1, ax1 = plt.subplots()
sns.barplot(x=metric_names, y=metric_values, palette='pastel', ax=ax1)
ax1.set_ylim(0, 1)
ax1.set_ylabel("Skor")
for i, v in enumerate(metric_values):
    ax1.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
st.pyplot(fig1)

# Confusion matrix
st.markdown("#### 🧩 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Tidak", "Ya"], yticklabels=["Tidak", "Ya"])
ax2.set_xlabel("Prediksi")
ax2.set_ylabel("Aktual")
st.pyplot(fig2)

# Classification report
st.markdown("#### 📋 Laporan Klasifikasi")
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.background_gradient(cmap="YlOrBr"), use_container_width=True)

# Catatan akhir
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 15px;'>"
    "Model yang digunakan: <b>Logistic Regression</b>. "
    "Evaluasi dilakukan pada data uji sebesar 20%. Visualisasi dan analisis metrik digunakan untuk membantu interpretasi performa model."
    "</div>",
    unsafe_allow_html=True
)
