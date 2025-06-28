
import streamlit as st
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Hardcoded employee dataset for demonstration
data = {
    "Employee ID": [
        "OPC00425", "OPC00426", "OPC01066", "OPC00417", "OPC00418",
        "OPC00419", "OPC00420", "OPC00421", "OPC00422", "OPC00423",
        "OPC00424", "OPC00427", "OPC00428", "OPC00429", "OPC00430",
        "OPC00431", "OPC00432"
    ],
    "Employee Name": [
        "Mohamed Salih Eidam Adam", "Ibrahim Abdel Aziz Mohamed", "Ali Ibrahim Gulfan",
        "Sami Mohamed Elfatih Saad Ahmed", "Hassan Eltayeb Hassan Mohammed",
        "Omer Ibrahim Omer", "Mansour Adam Mansour", "Ahmed Hassan Khalil",
        "Tariq Omer Ahmed", "Yasir Elhadi Mohammed", "Nour Eldin Ibrahim",
        "Osman Ahmed Osman", "Abdelrahman Ismail", "Salah Ahmed Abdelrahman",
        "Hassan Idris Ali", "Mohamed Ahmed Elamin", "Osman Khalid Bashir"
    ],
    "Position": [
        "Geological Superintendent", "Geological Superintendent", "Wellsite Geological Supervisor",
        "Wellsite Geological Supervisor", "Wellsite Geological Supervisor", "Wellsite Geological Supervisor",
        "Wellsite Geological Supervisor", "Wellsite Geological Supervisor", "Wellsite Geological Supervisor",
        "Operations Geologist", "Operations Geologist", "Pore Pressure Engineer", "Pore Pressure Engineer",
        "Wellsite Geological Supervisor", "Wellsite Geological Supervisor", "Geological Superintendent",
        "Geological Superintendent"
    ],
    "Email": [
        "meidam@2bopco.com", "iaziz@2bopco.com", "agulfan@gnpoc.com", "smelfatih@2bopco.com",
        "hehassan@2bopco.com", "oiomer@2bopco.com", "mamansour@2bopco.com", "ahkhalil@2bopco.com",
        "toahmed@2bopco.com", "yemohammed@2bopco.com", "neibrahim@2bopco.com", "oaosman@2bopco.com",
        "aismail@2bopco.com", "saahmed@2bopco.com", "hiali@2bopco.com", "maelamin@2bopco.com",
        "okbashir@2bopco.com"
    ]
}

df = pd.DataFrame(data)
df["Email Domain"] = df["Email"].apply(lambda x: x.split("@")[-1])
df["Name_Length"] = df["Employee Name"].apply(len)
df["ID_Num"] = df["Employee ID"].str.extract("(\d+)").astype(float)
df["Domain_ID"] = LabelEncoder().fit_transform(df["Email Domain"])
df["Position_Label"] = LabelEncoder().fit_transform(df["Position"])
label_encoder = LabelEncoder()
df["Position_Label"] = label_encoder.fit_transform(df["Position"])

# Train the model inside the app
features = df[["Name_Length", "ID_Num", "Domain_ID"]]
target = df["Position_Label"]
model = RandomForestClassifier(random_state=42)
model.fit(features, target)

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Geological Operations Assistant")

tab1, tab2, tab3, tab4 = st.tabs(["🧠 Role Predictor", "✉️ Email Validator", "🤖 Auto-fill Assistant", "📉 Attrition Predictor"])

with tab1:
    st.header("Role Prediction")
    name = st.text_input("Employee Name", "John Smith")
    emp_id = st.text_input("Employee ID", "EMP008")
    email = st.text_input("Email", "jsmith@2bopco.com")

    if st.button("Predict"):
        name_len = len(name)
        match = re.search(r"(\d+)", emp_id)
        id_num = float(match.group(1)) if match else 0.0
        domain_id = 0 if "2bopco.com" in email else 1
        features = [[name_len, id_num, domain_id]]
        pred_label = model.predict(features)[0]
        pred_role = label_encoder.inverse_transform([pred_label])[0]
        st.success(f"Predicted Role: **{pred_role}**")

    st.subheader("Position Distribution")
    st.bar_chart(df["Position"].value_counts())

with tab2:
    st.header("Email Domain Validator")
    email_input = st.text_input("Enter email to validate", "someone@example.com")
    if st.button("Validate Domain"):
        domain = email_input.split("@")[-1]
        valid = domain in df["Email Domain"].unique()
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
