import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)

# --- Load Model and Feature Names ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """Loads the saved model and feature names."""
    try:
        model = joblib.load('fraud_detection_pipeline.joblib')
        feature_names = joblib.load('feature_names.joblib')
        return model, feature_names
    except FileNotFoundError:
        return None, None

model_data = load_model()

# --- Main App ---
st.title("💳 Real-Time Fraud Detection System")
st.markdown("Enter transaction details on the left to get a fraud prediction.")

if model_data[0] is None:
    st.error(
        "**Model files not found!** 🚨\n"
        "Please run `fraud_pipeline_fixed.py` first to train and save the model."
    )
else:
    model, feature_names = model_data
    
    # --- Sidebar for Inputs ---
    st.sidebar.header("Input Transaction Features")
    
    # Create input fields
    # For a real app, you would have all 30. For this demo,
    # we'll create the most important ones and set others to 0.
    time = st.sidebar.number_input("Time (seconds since first transaction)", value=0.0)
    amount = st.sidebar.number_input("Transaction Amount ($)", value=100.0, format="%.2f")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Top V-Features (PCA)**")
    
    # Based on typical fraud model feature importance
    v4 = st.sidebar.slider("V4", -6.0, 17.0, 0.0)
    v10 = st.sidebar.slider("V10", -25.0, 24.0, 0.0)
    v12 = st.sidebar.slider("V12", -19.0, 8.0, 0.0)
    v14 = st.sidebar.slider("V14", -20.0, 11.0, 0.0)
    v17 = st.sidebar.slider("V17", -26.0, 10.0, 0.0)
    
    
    # --- Prediction Logic ---
    if st.button("Check Transaction", type="primary"):
        # 1. Create a dictionary of all features, default to 0.0
        input_data = {feature: 0.0 for feature in feature_names}
        
        # 2. Update the dictionary with user inputs
        input_data['Time'] = time
        input_data['Amount'] = amount
        input_data['V4'] = v4
        input_data['V10'] = v10
        input_data['V12'] = v12
        input_data['V14'] = v14
        input_data['V17'] = v17
        
        # 3. Convert to DataFrame (must be in the correct order)
        # The pipeline *requires* a DataFrame with these exact column names
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # 4. Make Prediction
        try:
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1] # Prob of fraud
            
            # 5. Display Result
            st.subheader("Prediction Result:")
            if prediction[0] == 1:
                st.error(f"**FRAUD DETECTED!** (Probability: {probability:.2%})", icon="🚨")
                st.warning("This transaction is highly suspicious and should be blocked.")
            else:
                st.success(f"**Normal Transaction** (Fraud Probability: {probability:.2%})", icon="✅")
                st.info("This transaction appears to be safe.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")