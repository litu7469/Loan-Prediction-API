import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('model_logistic_regression.pkl')
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", str(e))
    model = None

# Health check route
@app.route('/', methods=['GET'])
def home():
    return "Loan Prediction API is running"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json(force=True)
        print("📥 Received data:", data)

        features = data.get('features')
        if not isinstance(features, list):
            return jsonify({'error': 'Invalid input format. Expected a list of features.'}), 400

        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)

        print("📤 Prediction:", prediction[0])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

# Bind to dynamic port for Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Render sets this dynamically
    app.run(host='0.0.0.0', port=port)
