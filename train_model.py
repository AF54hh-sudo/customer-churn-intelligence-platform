import json 
import joblib

from datetime import datetime 

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier

from src.config import Config
from src.preprocessing import DataPreprocessor


def train_churn_model():
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data_preprocessor = DataPreprocessor()

    df = data_preprocessor.load_data()
    df = data_preprocessor.clean_data(df)

    X,y = data_preprocessor.split_features_target(df)

    preprocessing_pipeline, numeric_cols, categorical_cols=                                                       (data_preprocessor.build_preprocessing_pipeline(X))

    x_train, x_test, y_train,y_test = train_test_split(X,y, test_size=Config.TEST_SIZE, 
                                                       random_state=Config.RANDOM_STATE, stratify=y,
                                                      )

    x_train_processed=preprocessing_pipeline.fit_transform(x_train)
    x_test_processed = preprocessing_pipeline.transform(x_test)
                                                


        
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=Config.RANDOM_STATE,)

    model.fit(x_train_processed, y_train)

    y_pred =model.predict(x_test_processed)
    y_probability = model.predict_proba(x_test_processed)[:,1]


    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_probability), 4),
    }

    print("Model Training Completed")
    print("-" * 40)

    print("Model Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    feature_names = preprocessing_pipeline.get_feature_names_out().tolist()

    joblib.dump(model, Config.MODEL_PATH)
    joblib.dump(preprocessing_pipeline, Config.PREPROCESSOR_PATH)

    metadata = {
        "model_name": "XGBoost Classifier",
        "training_date": datetime.now().strftime("%Y-%m-%d"),
        "dataset": "IBM Telco Customer Churn Dataset",
        "target_column": Config.TARGET_COLUMN,
        "metrics": metrics,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "feature_names": feature_names,
    }

    with open(Config.METADATA_PATH, "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nSaved Files:")
    print(Config.MODEL_PATH)
    print(Config.PREPROCESSOR_PATH)
    print(Config.METADATA_PATH)

    return model, preprocessing_pipeline, metadata


if __name__ == "__main__":
    train_churn_model()