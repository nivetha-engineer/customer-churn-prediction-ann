import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Set page configuration
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# 1. Load the pipeline assets
@st.cache_resource # Caches the model so it doesn't reload on every click
def load_assets():
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    model = tf.keras.models.load_model("ann_churn_model.keras")
    return preprocessor, le, model

try:
    preprocessor, le, model = load_assets()
except Exception as e:
    st.error(f"Error loading model assets. Make sure 'preprocessor.pkl', 'label_encoder.pkl', and 'ann_churn_model.keras' are in the same folder as this script. Details: {e}")
    st.stop()

# Title and Description
st.title("🔮 Telco Customer Churn Prediction App")
st.markdown("Enter a customer's profile metrics below to calculate their risk of churning in real-time using our trained Artificial Neural Network.")
st.divider()

# Create layouts/columns for inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.header("👤 Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
    Partner = st.selectbox("Has a Partner?", ["Yes", "No"])
    Dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

with col2:
    st.header("📞 Services")
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col3:
    st.header("💳 Account & Billing")
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=65.0)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)

st.divider()

# Convert inputs into an exact dataframe structure matching training format
input_data = pd.DataFrame([{
    'gender': gender,
    'SeniorCitizen': 1 if SeniorCitizen == "Yes" else 0,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}])

# Prediction Trigger Button
if st.button("🔮 Calculate Churn Risk", type="primary"):
    # Preprocess incoming feature profile
    processed_input = preprocessor.transform(input_data)
    
    # Run the model forward pass (predict probabilities)
    predicted_prob = model.predict(processed_input)[0][0]
    
    # Use our optimal fine-tuned threshold we established (0.3022)
    OPTIMAL_THRESHOLD = 0.3022
    prediction_class = 1 if predicted_prob > OPTIMAL_THRESHOLD else 0
    
    # Map back to readable string using label encoder
    if isinstance(le, dict): # if mapping dict cleanup from earlier step was applied
        churn_status = le.get(prediction_class, "Unknown")
    else: # native sklearn label encoder
        churn_status = le.inverse_transform([prediction_class])[0]
        
    # Visual Output Display
    st.subheader("📊 Output Analysis")
    
    # Progress Bar metrics
    st.metric(label="Churn Probability Score", value=f"{predicted_prob:.2%}")
    st.progress(float(predicted_prob))
    
    if prediction_class == 1:
        st.error(f"⚠️ **High Churn Alert!** This customer falls above the safety threshold and is likely to **{churn_status}**.")
    else:
        st.success(f"✅ **Safe Zone!** This customer is steady and likely to remain **Retained**.")
