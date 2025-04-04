import streamlit as st
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Load trained model
model_path = "linear_regression_model.pkl"
scaler_path = "scaler.pkl"

try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
    st.stop()
except pickle.UnpicklingError:
    st.error("Error loading model. The file may be corrupted.")
    st.stop()

st.title("IBD Classification and Prediction")

# Input fields
feature_names = ["IBD_Type", "Family_History_IBD", "Diarrhea_Frequency", "Severity_Score"]
input_data = []

for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0)
    input_data.append(value)

# Convert to numpy array and reshape
input_data = np.array(input_data).reshape(1, -1)

# Ensure feature compatibility
if input_data.shape[1] != model.n_features_in_:
    st.error(f"Feature mismatch. Expected {model.n_features_in_} features but got {input_data.shape[1]}.")
    st.stop()

# Scale input data
input_array_scaled = scaler.transform(input_data)

if st.button("Predict IBD Type"):
    prediction = model.predict(input_array_scaled)[0]
    st.success(f"Predicted IBD Type: {prediction}")
