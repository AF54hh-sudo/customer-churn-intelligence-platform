import joblib
import pandas as pd

from src.config import Config


class ChurnPredictor:
    def __init__(self):
        self.model = joblib.load(Config.MODEL_PATH)
        self.preprocessor = joblib.load(Config.PREPROCESSOR_PATH)

    def get_risk_level(self, probability):
        if probability >= 0.70:
            return "High Risk"

        if probability >= 0.40:
            return "Medium Risk"

        return "Low Risk"

    def predict(self, customer_data):
        input_df = pd.DataFrame([customer_data])

        processed_input = self.preprocessor.transform(input_df)

        churn_probability = self.model.predict_proba(processed_input)[0][1]

        risk_level = self.get_risk_level(churn_probability)

        result = {
            "churn_probability": churn_probability,
            "churn_probability_percent": churn_probability * 100,
            "risk_level": risk_level,
        }

        return result