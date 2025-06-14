import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('jantung_model.pkl')

st.title("Prediksi Penyakit Jantung")

# Form input
with st.form("form_penyakit_jantung"):
    age = st.number_input('Usia', min_value=1, max_value=120)
    sex = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
    cp = st.selectbox('Tipe Nyeri Dada (Chest Pain Type)', [0, 1, 2, 3])
    trestbps = st.number_input('Tekanan Darah', min_value=80, max_value=200)
    chol = st.number_input('Kolesterol', min_value=100, max_value=600)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl?', ['Ya', 'Tidak'])
    restecg = st.selectbox('Hasil EKG saat istirahat', [0, 1, 2])
    thalach = st.number_input('Denyut Jantung Maksimum (thalach)', min_value=60, max_value=220)
    exang = st.selectbox('Angina saat olahraga?', ['Ya', 'Tidak'])
    oldpeak = st.number_input('Oldpeak (depresi ST)', min_value=0.0, max_value=6.0, step=0.1)
    slope = st.selectbox('Kemiringan segmen ST', [0, 1, 2])
    ca = st.selectbox('Jumlah pembuluh darah utama', [0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', [0, 1, 2, 3])

    submit = st.form_submit_button("Prediksi")

# Ketika tombol ditekan
if submit:
    features = np.array([[
        age,
        1 if sex == 'Laki-laki' else 0,  # Encoding for sex
        cp,
        trestbps,
        chol,
        1 if fbs == 'Ya' else 0,  # Encoding for fasting blood sugar
        restecg,
        thalach,
        1 if exang == 'Ya' else 0,  # Encoding for exercise angina
        oldpeak,
        slope,
        ca,
        thal
    ]])

    # Debugging: Print the shape of the features array
    st.write("Features shape:", features.shape)

    try:
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.error("Pasien berisiko terkena penyakit jantung.")
        else:
            st.success("Pasien tidak berisiko terkena penyakit jantung.")
    except ValueError as e:
        st.error(f"Error in prediction: {e}")
