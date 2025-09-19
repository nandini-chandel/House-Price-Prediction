from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
MODEL_PATH = "model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception:
    model = None

# Expected features based on dataset
EXPECTED_FEATURES = ['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'street', 'city', 'statezip', 'country']

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model is not available."}), 500

    try:
        data = request.get_json()
        # Ensure all expected features are provided
        features = [data.get(col, 0) for col in EXPECTED_FEATURES]
        prediction = model.predict([features])
        return jsonify({"prediction": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def home():
    return "House Price Prediction API is running! Expected features: " + ", ".join(EXPECTED_FEATURES)

if __name__ == "__main__":
    app.run(debug=True)
