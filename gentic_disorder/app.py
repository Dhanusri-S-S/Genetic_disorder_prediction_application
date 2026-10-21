
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Genetic Disorder Classification",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genetic Disorder Risk Classification System")

st.write(
    "Educational ML prototype for preliminary genetic disorder "
    "category classification. This is not a medical diagnosis."
)

# -------------------------------
# File Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_INFO_PATH = os.path.join(BASE_DIR, "feature_info.json")

# IMPORTANT:
# Replace this URL with your deployed Flask backend URL later.
API_URL = "http://127.0.0.1:5000/predict"

# -------------------------------
# Load Feature Information
# -------------------------------
try:
    with open(FEATURE_INFO_PATH, "r") as f:
        feature_info = json.load(f)
except FileNotFoundError:
    st.error(
        "feature_info.json was not found. "
        "Make sure it is inside the same folder as app.py."
    )
    st.stop()
except json.JSONDecodeError:
    st.error("feature_info.json contains invalid JSON.")
    st.stop()

# -------------------------------
# Patient Information Form
# -------------------------------
st.header("Patient Information")
st.write("Enter the available patient attributes below.")

with st.form("patient_form"):
    patient_data = {}

    for feature, info in feature_info.items():

        if info.get("type") == "numeric":
            default = info.get("default", 0.0)

            patient_data[feature] = st.number_input(
                feature,
                value=float(default),
                key=feature
            )

        else:
            options = info.get("options", [])

            if options:
                patient_data[feature] = st.selectbox(
                    feature,
                    options=options,
                    key=feature
                )
            else:
                patient_data[feature] = st.text_input(
                    feature,
                    key=feature
                )

    submitted = st.form_submit_button(
        "Predict Disorder Category"
    )

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    try:
        response = requests.post(
            API_URL,
            json=patient_data,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()

            st.subheader("Prediction Result")
            st.success(result["predicted_category"])

            probabilities = result.get("probabilities", {})

            if probabilities:
                st.subheader("Class Probability Estimates")

                probability_df = pd.DataFrame({
                    "Category": list(probabilities.keys()),
                    "Probability": list(probabilities.values())
                })

                probability_df["Percentage"] = (
                    probability_df["Probability"] * 100
                )

                st.bar_chart(
                    probability_df.set_index("Category")["Percentage"]
                )

                st.dataframe(
                    probability_df[["Category", "Percentage"]],
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "These are model estimates and may not represent "
                    "calibrated clinical risk values."
                )

        else:
            st.error(
                f"API error ({response.status_code}): "
                f"{response.text}"
            )

    except requests.exceptions.RequestException as e:
        st.error(
            "Could not connect to the prediction API. "
            "Check that the Flask backend is deployed and API_URL "
            "contains its public URL."
        )
        st.code(str(e))

# -------------------------------
# Model Performance Information
# -------------------------------
st.divider()
st.subheader("Model Performance")

st.write(
    "Refer to the validation metrics and confusion matrix "
    "from the model evaluation notebook."
)

st.info(
    "Accuracy alone is not enough to evaluate classification. "
    "Precision, recall, and F1-score for each category should "
    "also be considered."
)

st.caption(
    "This application is an educational prototype, "
    "not a diagnostic tool."
)
