import pandas as pd
import shap


class ChurnExplainer:
    def __init__(self, model, preprocessor, feature_names):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)

    def explain_prediction(self, customer_data, top_n=5):
        input_df = pd.DataFrame([customer_data])

        processed_input = self.preprocessor.transform(input_df)

        shap_values = self.explainer.shap_values(processed_input)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        single_prediction_values = shap_values[0]

        explanation_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "shap_value": single_prediction_values,
            }
        )

        explanation_df["absolute_impact"] = explanation_df["shap_value"].abs()

        explanation_df["direction"] = explanation_df["shap_value"].apply(
            lambda value: "Increases churn risk" if value > 0 else "Reduces churn risk"
        )

        explanation_df["readable_feature"] = explanation_df["feature"].apply(
            self.clean_feature_name
        )

        explanation_df = explanation_df.sort_values(
            by="absolute_impact",
            ascending=False,
        ).head(top_n)

        return explanation_df

    def clean_feature_name(self, feature_name):
        cleaned_name = (
            feature_name
            .replace("numeric__", "")
            .replace("categorical__", "")
        )

        known_columns = [
            "Payment Method",
            "Internet Service",
            "Online Security",
            "Online Backup",
            "Device Protection",
            "Tech Support",
            "Streaming TV",
            "Streaming Movies",
            "Phone Service",
            "Multiple Lines",
            "Paperless Billing",
            "Senior Citizen",
            "Tenure Months",
            "Monthly Charges",
            "Total Charges",
            "Zip Code",
        ]

        for column in known_columns:
            prefix = f"{column}_"

            if cleaned_name.startswith(prefix):
                value = cleaned_name.replace(prefix, "")
                return f"{column}: {value}"

        return cleaned_name

    def create_plain_english_reasons(self, explanation_df):
        reasons = []

        for _, row in explanation_df.iterrows():
            readable_feature = row["readable_feature"]
            direction = row["direction"]

            reason = f"{readable_feature} - {direction}"
            reasons.append(reason)

        return reasons