
import os
import json
import gzip
import joblib
import pandas as pd
import streamlit as st

# -----------------------------------
# Page configuration
# -----------------------------------
st.set_page_config(
    page_title="Genetic Disorder Classification",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genetic Disorder Risk Classification System")

st.write(
    "An educational machine learning prototype that predicts "
    "a genetic disorder category from the entered features."
)

st.warning(
    "This is an educational prototype, not a medical diagnosis. "
    "Prediction percentages are model estimates, not validated "
    "medical risk percentages."
)

# -----------------------------------
# File paths
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "genetic_disorder_model.pkl.gz"
)

FEATURE_PATH = os.path.join(
    BASE_DIR, "feature_info.json"
)

# -----------------------------------
# Load model
# -----------------------------------
@st.cache_resource
def load_model():
    with gzip.open(MODEL_PATH, "rb") as file:
        return joblib.load(file)

# -----------------------------------
# Load feature information and model
# -----------------------------------
try:
    with open(FEATURE_PATH, "r", encoding="utf-8") as file:
        feature_info = json.load(file)

    model = load_model()

except Exception as error:
    st.error("Could not load the model or feature information.")
    st.code(str(error))
    st.info(
        "Please verify that genetic_disorder_model.pkl.gz "
        "is a valid compressed model file and that feature_info.json "
        "is present in the same folder."
    )
    st.stop()

if not isinstance(feature_info, dict):
    st.error(
        "feature_info.json must contain a dictionary of features."
    )
    st.stop()

# -----------------------------------
# Model feature names
# -----------------------------------
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
else:
    model_features = list(feature_info.keys())

missing_info = [
    feature for feature in model_features
    if feature not in feature_info
]

if missing_info:
    st.error("Some model features are missing from feature_info.json.")
    st.write(missing_info)
    st.stop()

# -----------------------------------
# Patient information form
# -----------------------------------
st.header("Patient Information")
st.caption("Enter the available information and click Predict.")

with st.form("patient_form"):
    patient_data = {}

    for feature in model_features:
        info = feature_info[feature]

        field_type = info.get("type", "categorical")
        default = info.get("default", 0)

        if field_type == "numeric":
            try:
                default_value = float(default)
            except (TypeError, ValueError):
                default_value = 0.0

            patient_data[feature] = st.number_input(
                feature,
                value=default_value,
                key=feature
            )

        else:
            options = info.get("options", [])

            if options:
                default_value = info.get("default", options[0])

                default_index = (
                    options.index(default_value)
                    if default_value in options
                    else 0
                )

                patient_data[feature] = st.selectbox(
                    feature,
                    options=options,
                    index=default_index,
                    key=feature
                )

            else:
                patient_data[feature] = st.text_input(
                    feature,
                    value="" if default is None else str(default),
                    key=feature
                )

    submitted = st.form_submit_button(
        "Predict Disorder Category",
        type="primary",
        use_container_width=True
    )

# -----------------------------------
# Make prediction
# -----------------------------------
if submitted:
    try:
        # Preserve the original categorical values.
        # The trained Pipeline handles preprocessing.
        input_df = pd.DataFrame(
            [
                {
                    feature: patient_data[feature]
                    for feature in model_features
                }
            ],
            columns=model_features
        )

        prediction = model.predict(input_df)[0]
        predicted_category = str(prediction)

        st.divider()
        st.header("Prediction Result")
        st.success(f"Predicted category: {predicted_category}")

        # -----------------------------------
        # Display estimated probabilities
        # -----------------------------------
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            model_classes = model.classes_

            results = []

            for model_class, probability in zip(
                model_classes, probabilities
            ):
                results.append({
                    "Category": str(model_class),
                    "Estimated probability (%)": round(
                        float(probability) * 100, 2
                    )
                })

            results_df = pd.DataFrame(results)

            results_df = results_df.sort_values(
                "Estimated probability (%)",
                ascending=False
            )

            top_result = results_df.iloc[0]

            st.subheader("Prediction Probability")

            st.metric(
                label="Highest estimated probability",
                value=(
                    f"{top_result['Estimated probability (%)']:.2f}%"
                )
            )

            st.bar_chart(
                results_df.set_index("Category")[
                    "Estimated probability (%)"
                ]
            )

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Probabilities represent the model's estimates for its "
                "classes. They are not clinically validated individual "
                "risk percentages."
            )

        else:
            st.info(
                "This model does not provide predict_proba(). "
                "Only the predicted category is displayed."
            )

    except Exception as error:
        st.error("Prediction failed.")
        st.code(str(error))

        st.info(
            "Check that the input feature names, data types, and "
            "categorical values match those used during model training."
        )

# -----------------------------------
# Footer
# -----------------------------------
st.divider()
st.subheader("About this project")

st.write(
    "This application demonstrates machine learning-based "
    "classification using a trained Extra Trees model."
)

st.caption(
    "For learning and demonstration only. "
    "Do not use this application to make medical decisions."
)
