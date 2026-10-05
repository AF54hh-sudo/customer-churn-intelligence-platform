import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from predict import ChurnPredictor
from src.config import Config
from src.explainability import ChurnExplainer
from src.model_utils import (
    explain_risk_level,
    format_probability,
    generate_retention_recommendation,
    get_risk_color,
)


st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_app_artifacts():
    model = joblib.load(Config.MODEL_PATH)
    preprocessor = joblib.load(Config.PREPROCESSOR_PATH)

    with open(Config.METADATA_PATH, "r") as file:
        metadata = json.load(file)

    predictor = ChurnPredictor()

    explainer = ChurnExplainer(
        model=model,
        preprocessor=preprocessor,
        feature_names=metadata["feature_names"],
    )

    return model, preprocessor, metadata, predictor, explainer


def render_risk_badge(risk_level, risk_color):
    st.markdown(
        f"""
        <div style="
            display: inline-block;
            padding: 8px 14px;
            border-radius: 6px;
            color: white;
            background-color: {risk_color};
            font-weight: 700;
            margin-bottom: 12px;
        ">
            {risk_level}
        </div>
        """,
        unsafe_allow_html=True,
    )


model, preprocessor, metadata, predictor, explainer = load_app_artifacts()


st.title("Customer Churn Intelligence Platform")

st.caption(
    "A machine learning decision-support app for predicting customer churn risk "
    "with explainable AI and retention recommendations."
)


with st.sidebar:
    st.header("Customer Input")

    with st.expander("Demographics", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with st.expander("Account Details", expanded=True):
        tenure_months = st.slider(
            "Tenure Months",
            min_value=0,
            max_value=72,
            value=12,
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with st.expander("Services", expanded=True):
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"],
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"],
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"],
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"],
        )

    with st.expander("Billing and Value", expanded=True):
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=85.0,
            step=1.0,
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=1020.0,
            step=50.0,
        )

        cltv = st.number_input(
            "Customer Lifetime Value",
            min_value=0,
            max_value=10000,
            value=3500,
            step=100,
        )

    with st.expander("Location", expanded=False):
        city = st.text_input("City", "Los Angeles")

        zip_code = st.number_input(
            "Zip Code",
            min_value=90000,
            max_value=96162,
            value=90001,
        )

        latitude = st.number_input("Latitude", value=34.05)

        longitude = st.number_input("Longitude", value=-118.24)

    predict_button = st.button("Predict Churn Risk", use_container_width=True)


customer_data = {
    "City": city,
    "Zip Code": zip_code,
    "Latitude": latitude,
    "Longitude": longitude,
    "Gender": gender,
    "Senior Citizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure Months": tenure_months,
    "Phone Service": phone_service,
    "Multiple Lines": multiple_lines,
    "Internet Service": internet_service,
    "Online Security": online_security,
    "Online Backup": online_backup,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies,
    "Contract": contract,
    "Paperless Billing": paperless_billing,
    "Payment Method": payment_method,
    "Monthly Charges": monthly_charges,
    "Total Charges": total_charges,
    "CLTV": cltv,
}


if predict_button:
    prediction_result = predictor.predict(customer_data)

    churn_probability = prediction_result["churn_probability"]
    risk_level = prediction_result["risk_level"]

    risk_color = get_risk_color(risk_level)
    risk_explanation = explain_risk_level(risk_level)

    recommendation = generate_retention_recommendation(
        probability=churn_probability,
        cltv=cltv,
    )

    st.subheader("Prediction Summary")
    render_risk_badge(risk_level, risk_color)

    col1, col2, col3 = st.columns(3)

    col1.metric("Churn Probability", format_probability(churn_probability))
    col2.metric("Risk Level", risk_level)
    col3.metric("Customer Lifetime Value", f"${cltv:,.0f}")

    st.markdown(
        f"""
        <div style="
            padding: 16px;
            border-radius: 8px;
            border-left: 6px solid {risk_color};
            background-color: #f8f9fa;
            margin-top: 8px;
        ">
            <h4 style="margin-bottom: 6px;">Risk Interpretation</h4>
            <p style="margin: 0;">{risk_explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Retention Recommendation")
    st.info(recommendation)

    st.subheader("Top Prediction Drivers")

    explanation_df = explainer.explain_prediction(
        customer_data=customer_data,
        top_n=5,
    )

    plain_english_reasons = explainer.create_plain_english_reasons(
        explanation_df
    )

    for reason in plain_english_reasons:
        st.write(f"- {reason}")

    chart_df = explanation_df.sort_values("shap_value")

    fig, ax = plt.subplots(figsize=(9, 4.5))

    colors = chart_df["shap_value"].apply(
        lambda value: "#d62728" if value > 0 else "#2ca02c"
    )

    ax.barh(
        chart_df["readable_feature"],
        chart_df["shap_value"],
        color=colors,
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value")
    ax.set_title("Top Factors Influencing This Prediction")

    st.pyplot(fig)

    with st.expander("Detailed SHAP Values"):
        st.dataframe(
            explanation_df[
                [
                    "readable_feature",
                    "shap_value",
                    "direction",
                    "absolute_impact",
                ]
            ],
            use_container_width=True,
        )

    with st.expander("Customer Input Used By Model"):
        st.dataframe(pd.DataFrame([customer_data]), use_container_width=True)

else:
    st.info("Enter customer details in the sidebar and click Predict Churn Risk.")


st.divider()

st.subheader("Model Performance")

metrics = metadata["metrics"]

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

metric_col1.metric("Accuracy", metrics["accuracy"])
metric_col2.metric("Precision", metrics["precision"])
metric_col3.metric("Recall", metrics["recall"])
metric_col4.metric("F1 Score", metrics["f1_score"])
metric_col5.metric("ROC-AUC", metrics["roc_auc"])

with st.expander("Model Details"):
    st.write(f"Model: {metadata['model_name']}")
    st.write(f"Training Date: {metadata['training_date']}")
    st.write(f"Dataset: {metadata['dataset']}")
    st.write(f"Target Column: {metadata['target_column']}")