import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('model_logistic_regression.pkl')

# Health check route
@app.route('/', methods=['GET'])
def home():
    return "Loan Prediction API is running"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = data.get('features')

        if not isinstance(features, list):
            return jsonify({'error': 'Invalid input format. Expected a list of features.'}), 400

        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)

        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app on the dynamic port assigned by Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
