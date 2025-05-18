from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Declare model globally (not loaded here)
model = None

# ✅ Lazy loading the model only once on first request
@app.before_first_request
def load_model_once():
    global model
    if model is None:
        model = load_model("my_model/my_model.keras")
        print("Model loaded.")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # features = np.array(data['features']).reshape(1, -1)
    features = np.array(data['features'])

    # Validate shape
    if features.shape != (1, 20, 14):
        return jsonify({"error": f"Invalid input shape: expected (1, 20, 14), got {features.shape}"}), 400

    # Expand to batch shape (1, 20, 14)
    input_array = np.expand_dims(features, axis=0)

    # Predict
    prediction = model.predict(features)[0][0]
    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # For Render or local
    app.run(host='0.0.0.0', port=port, debug=True)