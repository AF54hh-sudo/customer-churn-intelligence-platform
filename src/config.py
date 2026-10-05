from pathlib import Path 

class Config:

    BASE_DIR = Path(__file__).resolve().parent.parent

    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"

    DATA_PATH = DATA_DIR / "Telco_customer_churn.xlsx"

    MODEL_PATH = MODEL_DIR / "churn_model.pkl"
    PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
    METADATA_PATH = MODEL_DIR / "model_metadata.json"

    TARGET_COLUMN = "Churn Value"

    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    