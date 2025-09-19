import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Expected features
EXPECTED_FEATURES = model.named_steps["preprocessor"].transformers_[0][2] + \
                    model.named_steps["preprocessor"].transformers_[1][2]

st.title("🏡 House Price Prediction App")
st.write("Enter the values for the following features to predict house price:")

# Create input fields dynamically
input_data = []
for feature in EXPECTED_FEATURES:
    val = st.text_input(f"Enter value for {feature}:", "")
    input_data.append(val)

# Convert input to numeric where possible
def safe_convert(val):
    try:
        return float(val)
    except:
        return val

input_data = [safe_convert(x) for x in input_data]

if st.button("Predict"):
    try:
        df = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)
        prediction = model.predict(df)[0]
        st.success(f"💰 Predicted House Price: {prediction:,.2f} USD")
    except Exception as e:
        st.error(f"Error: {e}")
