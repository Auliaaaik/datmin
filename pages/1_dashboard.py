import streamlit as st

st.set_page_config(page_title="Prediction")
st.title("Dashboard")

#Membaca dataset
df = pd.read_csv("model/Gagal_jantung.csv", sep=';')
  # Tampilkan dataframe
    st.subheader("Isi Dataset:")
    st.dataframe(df)

    # Info ringkas
    st.subheader("Info Statistik:")
    st.write(df.describe())
