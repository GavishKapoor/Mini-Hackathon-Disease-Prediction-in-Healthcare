# src/pipelines/prediction_pipeline.py
import pandas as pd
import joblib
import os
import sys
from src.exception import CustomException
from src.logger import logging

class PredictionPipeline:
    def __init__(self):
        try:
            self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            self.model_path = os.path.join("artifacts", "model.pkl")
            self.train_columns_path = os.path.join("artifacts", "train_columns.pkl")

            # Load artifacts
            self.preprocessor = joblib.load(self.preprocessor_path)  # expected dict {col: LabelEncoder}
            self.model = joblib.load(self.model_path)
            self.train_columns = joblib.load(self.train_columns_path)
            logging.info("Preprocessor, Model, and Train Columns loaded successfully")

        except Exception as e:
            logging.error("Error loading artifacts", exc_info=True)
            raise CustomException(e, sys)

    def _apply_label_encoders(self, df: pd.DataFrame):
        """Apply saved LabelEncoder dict to columns in df (inplace)."""
        preproc = self.preprocessor
        # If preprocessor is a dict of LabelEncoders -> apply them
        if isinstance(preproc, dict):
            for col, le in preproc.items():
                if col in df.columns:
                    # fill missing and map unseen to a safe value
                    df[col] = df[col].fillna("missing")
                    # For unseen categories, map to the nearest known class (we map to first class)
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df[col] = le.transform(df[col])
        else:
            # If preprocessor is a sklearn transformer supporting transform
            df_transformed = preproc.transform(df)
            return df_transformed
        return df

    def predict(self, input_df: pd.DataFrame):
        try:
            df = input_df.copy()
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            # Add missing columns with default 0
            missing_cols = set(self.train_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0

            # Keep only training columns in the correct order
            df = df[self.train_columns]

            # Apply encoders / preprocessor
            processed = self._apply_label_encoders(df)

            # If _apply_label_encoders returned an array (transformer path), use it
            if not isinstance(processed, pd.DataFrame):
                X = processed
            else:
                X = processed.values

            # Predict
            preds = self.model.predict(X)
            return preds

        except Exception as e:
            logging.error("Error during prediction", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    # quick local test
    test_file = "notebooks/data/testing.csv"
    df = pd.read_csv(test_file)
    pipeline = PredictionPipeline()
    preds = pipeline.predict(df)
    print("Predictions:", preds)
