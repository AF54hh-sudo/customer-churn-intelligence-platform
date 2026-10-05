# Dataset

This project uses the IBM Telco Customer Churn dataset from Kaggle.

Dataset link:
https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset

The dataset contains customer demographic, account, service, billing, location, churn, and customer lifetime value information for a fictional telecom company.

## Target Column

The model predicts:

`Churn Value`

Where:

- `0` means the customer did not churn
- `1` means the customer churned

## Leakage Columns Removed

The following columns are removed before model training:

- `Churn Label`
- `Churn Score`
- `Churn Reason`

`Churn Score` is removed because it is already a predictive score and would cause target leakage.

