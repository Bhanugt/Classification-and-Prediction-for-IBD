import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained model
with open("linear_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load scaler if used during training
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("IBD Classification and Prediction")

# Input fields
feature_names = ["IBD_Type", "Family_History_IBD", "Diarrhea_Frequency", "Severity_Score"]
input_data = []

for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0)
    input_data.append(value)

# Convert to numpy array and reshape
input_data = np.array(input_data).reshape(1, -1)  # Ensure correct shape

print("Expected number of features:", model.n_features_in_)
print("Shape of input_data:", input_data.shape)

input_array_scaled = scaler.transform(input_data)

if st.button("Predict IBD Type"):
    prediction = model.predict(input_array_scaled)[0]
    st.success(f"Predicted IBD Type: {prediction}")
