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
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': '🚀 Loan Prediction API is live!',
        'model_loaded': model is not None,
        'endpoints': {
            '/': 'GET - Health check',
            '/predict': 'POST - Make predictions'
        }
    })

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
    
    try:
        # Get JSON data
        data = request.get_json(force=True)
        print("📥 Incoming request data:", data)
        
        # Extract features
        features = data.get('features')
        if not features:
            return jsonify({
                'status': 'error',
                'message': 'Missing "features" field in request'
            }), 400
        
        if not isinstance(features, list):
            return jsonify({
                'status': 'error',
                'message': 'Invalid input format. Expected a list of features.'
            }), 400
        
        # Convert to numpy array
        input_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)
        
        # Get probability if available
        try:
            probability = model.predict_proba(input_array)
            prob_dict = {
                'rejected': float(probability[0][0]),
                'approved': float(probability[0][1])
            }
        except:
            prob_dict = None
        
        result = {
            'status': 'success',
            'prediction': int(prediction[0]),
            'loan_status': 'Approved' if prediction[0] == 1 else 'Rejected'
        }
        
        if prob_dict:
            result['probability'] = prob_dict
        
        print("📤 Prediction result:", result)
        return jsonify(result)
    
    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Test endpoint to verify API is working
@app.route('/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'API is working correctly',
        'example_request': {
            'url': '/predict',
            'method': 'POST',
            'body': {
                'features': [1, 1, 0, 1, 0, 5000, 1500, 120, 360, 1.0, 2]
            }
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

# Entry point for local development
# On Render, gunicorn will be used instead
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)