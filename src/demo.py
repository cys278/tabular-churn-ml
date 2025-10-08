import streamlit as st
import requests

st.title("📊 Customer Churn Predictor")

# Input fields
customer = {
    "customerID": st.text_input("Customer ID", "1234-ABCD"),
    "gender": st.selectbox("Gender", ["Male", "Female"]),
    "SeniorCitizen": st.selectbox("Senior Citizen", [0, 1]),
    "Partner": st.selectbox("Partner", ["Yes", "No"]),
    "Dependents": st.selectbox("Dependents", ["Yes", "No"]),
    "tenure": st.number_input("Tenure (months)", 1, 100, 5),
    "PhoneService": st.selectbox("Phone Service", ["Yes", "No"]),
    "MultipleLines": st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"]),
    "InternetService": st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    "OnlineSecurity": st.selectbox("Online Security", ["Yes", "No", "No internet service"]),
    "OnlineBackup": st.selectbox("Online Backup", ["Yes", "No", "No internet service"]),
    "DeviceProtection": st.selectbox("Device Protection", ["Yes", "No", "No internet service"]),
    "TechSupport": st.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
    "StreamingTV": st.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
    "StreamingMovies": st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"]),
    "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": st.selectbox("Paperless Billing", ["Yes", "No"]),
    "PaymentMethod": st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]),
    "MonthlyCharges": st.number_input("Monthly Charges", 0.0, 200.0, 89.5),
    "TotalCharges": st.number_input("Total Charges", 0.0, 10000.0, 445.0),
}


if st.button("Predict Churn"):
    response = requests.post("http://127.0.0.1:8000/predict", json=customer)
    st.json(response.json())