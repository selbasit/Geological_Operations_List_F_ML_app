
import streamlit as st
import pandas as pd
import joblib
import re

model = joblib.load("role_predictor_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def extract_features(name, emp_id, email):
    name_len = len(name)
    match = re.search(r"(\d+)", emp_id)
    id_num = float(match.group(1)) if match else 0.0
    domain_id = 0 if "2bopco.com" in email else 1
    return [[name_len, id_num, domain_id]]

st.set_page_config(layout="wide")
st.title("Geological Operations Assistant")

tab1, tab2, tab3, tab4 = st.tabs(["🧠 Role Predictor", "✉️ Email Validator", "🤖 Auto-fill Assistant", "📉 Attrition Predictor"])

with tab1:
    st.header("Role Prediction")
    name = st.text_input("Employee Name", "John Smith")
    emp_id = st.text_input("Employee ID", "EMP008")
    email = st.text_input("Email", "jsmith@2bopco.com")

    if st.button("Predict"):
        features = extract_features(name, emp_id, email)
        pred_label = model.predict(features)[0]
        pred_role = label_encoder.inverse_transform([pred_label])[0]
        st.success(f"Predicted Role: **{pred_role}**")

    st.subheader("Position Distribution (Simulated)")
    role_counts = pd.DataFrame({
        "Position": ["Wellsite Geological Supervisor", "Geological Superintendent", "Operations Geologist", "Pore Pressure Engineer"],
        "Count": [8, 5, 2, 2]
    })
    st.bar_chart(role_counts.set_index("Position"))

with tab2:
    st.header("Email Domain Validator")
    email_input = st.text_input("Enter email to validate", "someone@example.com")
    if st.button("Validate Domain"):
        domain = email_input.split("@")[-1]
        valid = domain in ["2bopco.com", "gnpoc.com"]
        st.success("Valid domain.") if valid else st.error("Unknown or suspicious domain.")

with tab3:
    st.header("Smart Auto-fill Assistant")
    partial_id = st.text_input("Start typing Employee ID", "OPC")
    if partial_id.startswith("OPC"):
        st.info("Auto-suggestion: ID format matches company pattern.")
        suggested_email = "user@2bopco.com"
        st.text(f"Suggested Email: {suggested_email}")

with tab4:
    st.header("Attrition Predictor (Coming Soon)")
    st.warning("This module will use additional HR features to predict retention risk.")
