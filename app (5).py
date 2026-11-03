%%writefile app.py
import os
import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Loan Default Risk Decision Support", 
    page_icon="🏦", 
    layout="centered"
)

# Load model pipeline safely from model.pkl
@st.cache_resource
def load_pipeline():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

pipeline = load_pipeline()

st.title("🏦 Alpha Dreamers Banking Consortium")
st.subheader("Loan Officer Decision-Support Prototype")
st.markdown("This tool evaluates applicant default risk using a validated **Logistic Regression pipeline**[cite: 4].")

if pipeline is None:
    st.error("⚠️ Trained model file `model.pkl` not found! Please run your `train.py` script first.")
else:
    with st.form("loan_form"):
        st.markdown("### Applicant Financial Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.number_input("Annual Income ($)", min_value=10000.0, max_value=1000000.0, value=65000.0, step=1000.0)
            loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000.0, max_value=200000.0, value=15000.0, step=500.0)
        
        with col2:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700, step=1)
            employment_length = st.number_input("Employment Length (Years)", min_value=0, max_value=50, value=5, step=1)
        
        loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"])
        
        submitted = st.form_submit_button("Evaluate Default Risk")

    if submitted:
        input_data = pd.DataFrame({
            "income": [income],
            "loan_amount": [loan_amount],
            "credit_score": [credit_score],
            "employment_length": [employment_length],
            "loan_intent": [loan_intent]
        })
        
        prob = pipeline.predict_proba(input_data)[0][1]
        
        st.markdown("---")
        st.subheader("Evaluation Result")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(label="Estimated Default Probability", value=f"{prob:.2%}")
        with col_res2:
            risk_category = "High Risk (Review Required)" if prob > 0.35 else "Low Risk (Standard Processing)"
            st.metric(label="Recommended Risk Tier", value=risk_category)
            
        if prob > 0.35:
            st.warning("⚠️ **Flagged for Review**: Manual verification advised.")
        else:
            st.success("✅ **Standard Processing**: Applicant falls within acceptable parameters.")
