import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Risk Assessment")
st.write("Enter the patient's health information below.")

model_path = "models/diabetes_model.pkl"

if not os.path.exists(model_path):
    st.error("Model file not found!")
    st.info("Please run train_model.py first to create the model.")
    st.stop()

model = joblib.load(model_path)

st.subheader("Patient Information")

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0,
    max_value=300,
    value=120
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=80
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

if st.button("🔍 Predict Diabetes Risk"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
        st.write(f"Estimated diabetes probability: **{probability * 100:.2f}%**")