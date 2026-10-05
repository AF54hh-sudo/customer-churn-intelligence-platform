import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import Config


class DataPreprocessor:

    def __init__(self):

        self.target_column = Config.TARGET_COLUMN

        self.drop_columns = [
            "CustomerID",
            "Count",
            "Country",
            "State",
            "Lat Long",
            "Churn Label",
            "Churn Score",
            "Churn Reason",
        ]

    def load_data(self):
        """
        Load raw dataset from Excel file.
        """

        df = pd.read_excel(Config.DATA_PATH)

        return df

    def clean_data(self, df):
        """
        Perform dataset cleaning and preprocessing preparation.
        """

        df = df.copy()

        # Remove leading/trailing spaces from column names
        df.columns = df.columns.str.strip()

        # Convert Total Charges to numeric
        if "Total Charges" in df.columns:
            df["Total Charges"] = pd.to_numeric(
                df["Total Charges"],
                errors="coerce"
            )

        # Drop leakage and irrelevant columns
        columns_to_drop = [
            column
            for column in self.drop_columns
            if column in df.columns
        ]

        df = df.drop(columns=columns_to_drop)

        # Remove rows where target is missing
        df = df.dropna(
            subset=[self.target_column]
        )

        return df

    def split_features_target(self, df):
        """
        Split dataframe into X and y.
        """

        X = df.drop(
            columns=[self.target_column]
        )

        y = df[self.target_column].astype(int)

        return X, y

    def build_preprocessing_pipeline(self, X):
        """
        Create preprocessing pipeline.
        """

        categorical_cols = (
            X.select_dtypes(
                include=["object"]
            )
            .columns
            .tolist()
        )

        numeric_cols = (
            X.select_dtypes(
                include=["int64", "float64"]
            )
            .columns
            .tolist()
        )

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_cols
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_cols
                ),
            ]
        )

        return (
            preprocessor,
            numeric_cols,
            categorical_cols
        )