import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt

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

API_URL = "http://127.0.0.1:5000/predict"

with open("feature_info.json", "r") as f:
    feature_info = json.load(f)

st.header("Patient Information")
st.write("Enter the available patient attributes below.")

with st.form("patient_form"):
    patient_data = {}

    for feature, info in feature_info.items():
        if info["type"] == "numeric":
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

    submitted = st.form_submit_button("Predict Disorder Category")

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
                    "Probabilities are model estimates and may not be "
                    "calibrated clinical risk values."
                )

        else:
            st.error(f"API error ({response.status_code}): {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(
            "Could not connect to the Flask API. Make sure the API is running."
        )
        st.code(str(e))

st.divider()
st.subheader("Model Performance")

st.write(
    "Use the validation metrics and confusion matrix from your "
    "model evaluation notebook here. The values below should come "
    "from the same saved Logistic Regression model."
)

st.info(
    "Overall accuracy alone is not enough for classification. "
    "Compare per-class precision, recall and F1-score, especially "
    "because the three categories have different sample counts."
)
