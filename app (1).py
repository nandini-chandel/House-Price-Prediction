import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# --- Load model safely ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found! Expected at: {MODEL_PATH}")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# --- Detect expected features ---
try:
    numeric_features = model.named_steps["preprocessor"].transformers_[0][2]
    categorical_features = model.named_steps["preprocessor"].transformers_[1][2]
    EXPECTED_FEATURES = list(numeric_features) + list(categorical_features)
except Exception:
    EXPECTED_FEATURES = []

st.title("🏡 House Price Prediction App")
st.write("Predict house prices using your trained ML model!")

# --- Option 1: Manual input ---
st.header("Manual Input")
input_data = []
for feature in EXPECTED_FEATURES:
    val = st.text_input(f"Enter value for {feature}:", "")
    input_data.append(val)

def safe_convert(val):
    try:
        return float(val)
    except:
        return val

input_data = [safe_convert(x) for x in input_data]

if st.button("Predict from manual input"):
    try:
        df = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)
        prediction = model.predict(df)[0]
        st.success(f"💰 Predicted House Price: {prediction:,.2f} USD")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Option 2: CSV Upload ---
st.header("CSV Upload")
uploaded_file = st.file_uploader("Upload a CSV file with the same features", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview:", df.head())
        predictions = model.predict(df)
        df["PredictedPrice"] = predictions
        st.write("✅ Predictions:")
        st.dataframe(df)
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
