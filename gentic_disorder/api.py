
import os
import gzip
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "genetic_disorder_model.pkl.gz")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_info.json")

with gzip.open(MODEL_PATH, "rb") as file:
    model = joblib.load(file)

with open(FEATURE_PATH, "r") as file:
    feature_info = json.load(file)

feature_names = list(model.feature_names_in_)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Genetic Disorder Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        missing = [name for name in feature_names if name not in data]

        if missing:
            return jsonify({
                "error": "Missing input features",
                "missing_features": missing
            }), 400

        patient = pd.DataFrame(
            [[data[name] for name in feature_names]],
            columns=feature_names
        )

        prediction = model.predict(patient)[0]
        probabilities = model.predict_proba(patient)[0]

        probability_result = {
            str(label): round(float(prob) * 100, 2)
            for label, prob in zip(model.classes_, probabilities)
        }

        return jsonify({
            "predicted_category": str(prediction),
            "probabilities": probability_result,
            "model_confidence_percent": round(
                float(max(probabilities)) * 100, 2
            )
        })

    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )

