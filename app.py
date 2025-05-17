from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model("my_model/my_model.keras")  # Ensure this folder is included

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0][0]
    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)