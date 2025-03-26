import streamlit as st
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Load trained model
model_path = os.path.join(os.getcwd(), "linear_regression_model.pkl")

if os.path.exists(model_path):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
else:
    st.error("❌ Model file not found. Please upload `linear_regression_model.pkl`.")

# Load scaler if used during training
scaler_path = os.path.join(os.getcwd(), "scaler.pkl")

if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
else:
    st.error("❌ Scaler file not found. Please upload `scaler.pkl`.")

st.title("IBD Classification and Prediction")

# Input fields
feature_names = ["IBD_Type", "Family_History_IBD", "Diarrhea_Frequency", "Severity_Score"]
input_data = []

for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0)
    input_data.append(value)

# Convert to numpy array and reshape
input_data = np.array(input_data).reshape(1, -1)  # Ensure correct shape

# Display feature info
if 'model' in locals():
    st.write(f"✅ Expected number of features: {model.n_features_in_}")
    st.write(f"✅ Shape of input_data: {input_data.shape}")

# Scale input data
if 'scaler' in locals():
    input_array_scaled = scaler.transform(input_data)
else:
    input_array_scaled = input_data  # Use raw data if scaler is missing

# Prediction
if st.button("Predict IBD Type"):
    if 'model' in locals():
        prediction = model.predict(input_array_scaled)[0]
        st.success(f"🎯 Predicted IBD Type: {prediction}")
    else:
        st.error("❌ Model not loaded. Unable to make predictions.")
