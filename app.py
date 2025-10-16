import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model_logistic_regression.pkl'
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from '{MODEL_PATH}'")
except Exception as e:
    print(f"❌ Failed to load model from '{MODEL_PATH}': {e}")
    model = None

# Health check route
@app.route('/', methods=['GET'])
def health_check():
    return "🚀 Loan Prediction API is live!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json(force=True)
        print("📥 Incoming request data:", data)

        features = data.get('features')
        if not isinstance(features, list):
            return jsonify({'error': 'Invalid input format. Expected a list of features.'}), 400

        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)

        print("📤 Prediction result:", prediction[0])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({'error': str(e)}), 500

# Entry point for Render deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use dynamic port from environment
    app.run(host='0.0.0.0', port=port)
