# Customer Churn Intelligence Platform

A Streamlit machine learning app that predicts customer churn probability, explains the prediction with SHAP, and suggests retention actions based on customer risk and lifetime value.

## Project Overview

Customer churn is one of the most important business problems for subscription and telecom companies. This project uses the IBM Telco Customer Churn dataset to build an explainable churn prediction platform.

The app allows a user to enter customer details and receive:

- Churn probability
- Risk level
- Customer lifetime value context
- Retention recommendation
- SHAP explanation of the top churn drivers
- Model performance metrics

## Dataset

Dataset: IBM Telco Customer Churn Dataset

Source:
https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset

Target column:

`Churn Value`

Important leakage columns removed during training:

- `Churn Label`
- `Churn Score`
- `Churn Reason`

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Joblib
- OpenPyXL

## Model

Final model:

`XGBoost Classifier`

XGBoost was selected because it performs well on structured tabular data, captures non-linear patterns, handles feature interactions, and works well with SHAP explainability.

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.7977 |
| Precision | 0.6404 |
| Recall | 0.5428 |
| F1 Score | 0.5876 |
| ROC-AUC | 0.8512 |

## Project Structure

```text
customer-churn-intelligence-platform/
├── app.py
├── predict.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   ├── Telco_customer_churn.xlsx
│   └── README.md
├── models/
│   ├── churn_model.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.json
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── model_utils.py
│   └── explainability.py
└── notebook/
```

## How To Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

If Streamlit is not recognized, run:

```bash
python -m streamlit run app.py
```

## Key Features

- End-to-end ML pipeline
- Customer churn probability prediction
- Risk categorization
- SHAP explainability
- Retention recommendation logic
- Model metadata tracking
- Streamlit web interface

## Business Interpretation

The model helps identify customers who are likely to churn. In a real business setting, high-risk and high-value customers can be prioritized for retention campaigns, support outreach, loyalty offers, or contract upgrades.

Accuracy alone is not enough for churn prediction because the dataset is imbalanced. Recall, F1-score, and ROC-AUC are more useful for evaluating how well the model identifies churn-risk customers.

## Future Improvements

- Add threshold tuning to improve recall
- Compare Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost
- Add model monitoring and drift detection
- Add batch prediction from uploaded CSV files
- Deploy on Hugging Face Spaces
- Add screenshots and live demo link

