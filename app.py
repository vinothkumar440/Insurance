import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Load Models and Scaler ---
# Load the trained GradientBoostingRegressor model
try:
    model = joblib.load('gbmreg_model.pkl')
except FileNotFoundError:
    st.error("Model file 'lr_model.pkl' not found. Please make sure the model is trained and saved.")
    st.stop() # Stop the app if the model file is not found

# Load the scaler used during training
try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Scaler file 'scaler.pkl' not found. Please make sure the the scaler is saved after fitting.")
    st.stop() # Stop the app if the scaler file is not foun
# --- Streamlit App Layout ---
st.title('Insurance Charges Prediction')

st.write("""
Enter the details below to predict the insurance charges.
""")

# --- User Inputs ---
# Features: claim_amount, past_consultations, hospital_expenditure, annual_salary, children, smoker

claim_amount = st.number_input('Claim Amount', min_value=0.0, max_value=100000.0, value=25000.0)
past_consultations = st.slider('Number of Past Consultations', 0, 35, 10)
hospital_expenditure = st.number_input('Hospital Expenditure', min_value=0.0, max_value=50000000.0, value=5000000.0)
annual_salary = st.number_input('Annual Salary', min_value=0.0, max_value=5000000000.0, value=50000000.0)
children = st.slider('Number of Children', 0, 5, 0)
smoker_input = st.selectbox('Smoker', ['No', 'Yes'])

# --- Preprocessing User Inputs ---
# Convert smoker input to numerical (0 or 1) as per Label Encoding
smoker_encoded = 1 if smoker_input == 'Yes' else 0

# Create a DataFrame from user inputs
input_df = pd.DataFrame({
    'claim_amount': [claim_amount],
    'past_consultations': [past_consultations],
    'hospital_expenditure': [hospital_expenditure],
    'annual_salary': [annual_salary],
    'children': [float(children)], # Ensure children is float as it was in the processed data
    'smoker': [float(smoker_encoded)] # Ensure smoker is float
})


# --- Feature Scaling ---
# Apply the same scaling as done during training
# The scaler was fitted on all columns of 'x', so pass all relevant columns from input_df
# It's crucial that the input_df has the same columns in the same order as the training data 'x'

# Assuming 'x' had columns in this order: ['claim_amount', 'past_consultations', 'hospital_expenditure', 'annual_salary', 'children', 'smoker']
# (This should be verified from the training step)

# Ensure all columns passed to scaler.transform are the ones it was fitted on
# and in the correct order.
# A robust way is to save the column names from x during training and load them here.
# For now, we assume input_df's columns match x's columns and order.
input_df_scaled_array = scaler.transform(input_df)
input_df_scaled = pd.DataFrame(input_df_scaled_array, columns=input_df.columns)


# Ensure the input features are in the exact order the model was trained on
# The model.n_features_in_ attribute tells us how many features the model expects.
if model.n_features_in_ != input_df_scaled.shape[1]:
    st.error(f"Error: The number of input features ({input_df_scaled.shape[1]}) does not match the model's expected features ({model.n_features_in_}). Please check the input data preparation.")
    st.stop()

# --- Make Prediction ---
if st.button('Predict Charges'):
    try:
        prediction = model.predict(input_df_scaled)
        st.success(f'Predicted Insurance Charges: ${prediction[0]:,.2f}')
    except Exception as e:
        st.error(f"Error during prediction: {e}. Please check input values and model compatibility.")
