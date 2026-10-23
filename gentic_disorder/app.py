
import os
import json
import gzip
import joblib
import pandas as pd
import streamlit as st

# -----------------------------
# Page configuration
# -----------------------------
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

# -----------------------------
# Load files from this folder
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "genetic_disorder_model.pkl.gz"
)
FEATURE_PATH = os.path.join(
    BASE_DIR, "feature_info.json"
)

@st.cache_resource
def load_model():
    with gzip.open(MODEL_PATH, "rb") as file:
        return joblib.load(file)

try:
    with open(FEATURE_PATH, "r", encoding="utf-8") as file:
        feature_info = json.load(file)

    model = load_model()

except Exception as error:
    st.error("Could not load the model or feature information.")
    st.code(str(error))
    st.stop()

if not isinstance(feature_info, dict):
    st.error("feature_info.json must contain a feature dictionary.")
    st.stop()

# -----------------------------
# Disorder category names
# -----------------------------
class_names = {
    0: "Mitochondrial genetic inheritance disorders",
    1: "Multifactorial genetic inheritance disorders",
    2: "Single-gene inheritance diseases"
}

def get_class_name(value):
    # Handle numeric class labels such as 0, 1, 2
    if value in class_names:
        return class_names[value]

    # Also handle string labels such as "0", "1", "2"
    try:
        numeric_value = int(value)
        if numeric_value in class_names:
            return class_names[numeric_value]
    except (ValueError, TypeError):
        pass

    # Preserve class names if the model already returns text
    return str(value)

# -----------------------------
# Patient information form
# -----------------------------
st.header("Patient Information")
st.caption("Enter the available information and click Predict.")

with st.form("patient_form"):
    patient_data = {}

    for feature, info in feature_info.items():
        field_type = info.get("type", "categorical")
        default = info.get("default", 0)

        if field_type == "numeric":
            patient_data[feature] = st.number_input(
                feature,
                value=float(default),
                key=feature
            )

        else:
            options = info.get("options", [])

            if options:
                default_value = info.get("default", options[0])
                default_index = (
                    options.index(default_value)
                    if default_value in options else 0
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
                    value=str(default) if default is not None else "",
                    key=feature
                )

    submitted = st.form_submit_button(
        "Predict Disorder Category",
        type="primary",
        use_container_width=True
    )

# -----------------------------
# Make prediction
# -----------------------------
if submitted:
    try:
        # Use the model's original feature order when available
        if hasattr(model, "feature_names_in_"):
            model_features = list(model.feature_names_in_)
        else:
            model_features = list(feature_info.keys())

        missing_features = [
            feature for feature in model_features
            if feature not in patient_data
        ]

        if missing_features:
            st.error(
                "Some model features are missing from feature_info.json:"
            )
            st.write(missing_features)
            st.stop()

        # Convert categorical values to numeric codes.
        # This assumes the training notebook used LabelEncoder-style
        # alphabetical encoding for categorical input features.
        encoded_data = {}

        for feature in model_features:
            value = patient_data[feature]
            info = feature_info.get(feature, {})

            if info.get("type") == "numeric":
                encoded_data[feature] = float(value)

            else:
                options = info.get("options", [])

                if options:
                    sorted_options = sorted(
                        [str(option) for option in options]
                    )
                    encoded_data[feature] = (
                        sorted_options.index(str(value))
                    )
                else:
                    try:
                        encoded_data[feature] = float(value)
                    except (ValueError, TypeError):
                        st.error(
                            f"Cannot convert '{feature}' to a numeric "
                            "value. Its original training encoding is needed."
                        )
                        st.stop()

        input_df = pd.DataFrame(
            [encoded_data],
            columns=model_features
        )

        prediction = model.predict(input_df)[0]
        predicted_category = get_class_name(prediction)

        st.divider()
        st.header("Prediction Result")
        st.success(f"Predicted category: {predicted_category}")

        # Show estimated class probabilities if supported
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            model_classes = model.classes_

            results = []

            for model_class, probability in zip(
                model_classes, probabilities
            ):
                results.append({
                    "Category": get_class_name(model_class),
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
                value=f"{top_result['Estimated probability (%)']:.2f}%"
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
                "Probabilities indicate the model's relative estimates "
                "for its classes. They are not clinically validated "
                "individual risk percentages."
            )

        else:
            st.info(
                "This model does not provide predict_proba(). "
                "Only the predicted category can be displayed."
            )

    except Exception as error:
        st.error("Prediction failed. Please check the input encoding.")
        st.code(str(error))
        st.info(
            "The categorical encoding must exactly match the encoding "
            "used when training the model. If this error appears, "
            "we should use the original training encoders rather than "
            "guessing the category codes."
        )

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.subheader("About this project")
st.write(
    "This application demonstrates machine learning-based "
    "classification using a trained Extra Trees model."
)
st.caption(
    "For learning and demonstration only. Do not use this application "
    "to make medical decisions."
)
