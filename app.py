import streamlit as st
import pandas as pd
import joblib

st.title("💳 Credit Card Fraud Detection System")

# Load model and encoders
model = joblib.load("fraud_detection_model.joblib")
encoders = joblib.load("label_encoders.joblib")

# User Input Section
st.header("📥 Enter Transaction Details")

input_data = {}

# Numerical Inputs
for col in ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
    input_data[col] = st.number_input(f"{col}", min_value=0.0, format="%.2f")

# Transaction Type
type_input = st.selectbox("Transaction Type", ['CASH_OUT', 'PAYMENT', 'TRANSFER', 'DEBIT'])

try:
    input_data['type'] = encoders['type'].transform([type_input])[0]
except Exception as e:
    st.error(f"Encoding error: {e}")
    st.stop()

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Predict Button
if st.button("🚨 Check for Fraud"):
    try:
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.error("❌ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")
    except Exception as e:
        st.error(f"Prediction error: {e}")
