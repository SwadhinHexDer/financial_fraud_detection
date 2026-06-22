import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from src.utils import trigger_security_alert

st.set_page_config(page_title="Financial Fraud Watch Engine", layout="wide")

st.title("🛡️ Institutional Risk Scoring & Financial Fraud Detection System")
st.markdown("An operational interface evaluation layer backed by a hybrid XGBoost & Isolation Forest configuration framework.")

# Load internal runtime components
@st.cache_resource
def load_assets():
    xgb = XGBClassifier()
    xgb.load_model("saved_models/xgboost_model.json")
    iso = joblib.load("saved_models/isolation_forest.pkl")
    return xgb, iso

try:
    xgb, iso = load_assets()
    st.sidebar.success("✅ Engine components loaded successfully.")
except Exception as e:
    st.sidebar.error("❌ Model artifacts missing. Run pipeline first.")

# Create transaction data input form fields
st.subheader("Manual Transaction Verification Stream")
col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input("Transaction Volume ($)", min_value=0.01, value=250.00)
    tx_velocity = st.slider("Transaction Frequency Delta (Velocity Index)", 0.0, 1.0, 0.15)
with col2:
    time_window = st.number_index = st.number_input("Time Elapsed Since Last Session (Sec)", min_value=0, value=120)
    geo_consistency = st.slider("Geographic Dispersion Inconsistency Index", 0.0, 5.0, 0.42)
with col3:
    st.write("**Simulated Latent Dimension Vector (PCA Fields V1-V28)**")
    v1 = st.slider("Component Parameter V1 Variance", -5.0, 5.0, 0.0)
    v2 = st.slider("Component Parameter V2 Variance", -5.0, 5.0, 0.0)

# Structural assembly for prediction processing
if st.button("Evaluate Transaction Vector Risk Profiles"):
    # Rebuilding structural dimension array matching training features
    base_features = [time_window] + [v1, v2] + [0.0]*26 + [tx_velocity, geo_consistency, amount]
    feature_matrix = pd.DataFrame([base_features])
    
    # Evaluation execution
    risk_probability = xgb.predict_proba(feature_matrix)[0][1]
    anomaly_status = iso.predict(feature_matrix)[0]
    
    st.markdown("---")
    st.subheader("Engine Evaluation Results Metrics")
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="Calculated Statistical Fraud Vector Probability", value=f"{risk_probability:.2%}")
    with m2:
        status_label = "ANOMALOUS SEQUENCE DETECTED" if anomaly_status == -1 else "STANDARD ISOLATION ENVELOPE"
        st.metric(label="Isolation Forest Envelope Diagnostic", value=status_label)
        
    if risk_probability > 0.85:
        st.error("🚨 CRITICAL RESPONSE PROTOCOL: Highly Indicative Behavioral Signatures Detected.")
        ui_alert, mail_alert = trigger_security_alert({"Amount": amount, "Velocity": tx_velocity}, risk_probability)
        st.text_area("Triggered System Logging Operations Output:", value=mail_alert, height=180)
    else:
        st.success("✅ Operational Clearance Granted. Transaction fits standard consumer behavior matrices.")