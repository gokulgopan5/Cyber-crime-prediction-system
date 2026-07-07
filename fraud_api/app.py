from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load models
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')
encoder = joblib.load('model/encoder.pkl')

print("Scaler expects:", scaler.n_features_in_)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        amount = float(data['amount'])
        txn_type = data['type']
        oldbalance = float(data['oldbalanceOrg'])
        newbalance = float(data['newbalanceOrig'])

        # Encode
        txn_type_encoded = encoder.transform([txn_type])[0]

        # EXACT 4 FEATURES
        features = np.array([[amount, txn_type_encoded, oldbalance, newbalance]])

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]

        if prediction == -1:
            return jsonify({"prediction": "Fraud", "risk_score": 90})
        else:
            return jsonify({"prediction": "Normal", "risk_score": 10})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)