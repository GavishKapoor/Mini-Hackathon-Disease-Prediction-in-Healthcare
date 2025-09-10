# src/components/data_transformation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from src.logger import logging
from src.exception import CustomException
import os
import sys
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    train_columns_file_path: str = os.path.join('artifacts', 'train_columns.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Data Transformation method starts")
        try:
            # Read train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test data read successfully")

            # Drop unnamed or empty columns
            train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
            test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

            # Fill missing numerical values with median
            for col in train_df.select_dtypes(include=[np.number]).columns:
                median = train_df[col].median()
                train_df[col].fillna(median, inplace=True)
                test_df[col].fillna(median, inplace=True)

            # Encode categorical variables
            label_encoders = {}
            for col in train_df.select_dtypes(include=[object]).columns:
                # Skip columns that are fully empty
                if train_df[col].isnull().all():
                    continue

                le = LabelEncoder()
                # Fill missing categorical with string 'missing' before encoding
                train_df[col] = train_df[col].fillna("missing")
                test_df[col] = test_df[col].fillna("missing")

                le.fit(pd.concat([train_df[col], test_df[col]], axis=0))  # fit on union to reduce unseen later
                train_df[col] = le.transform(train_df[col])
                test_df[col] = le.transform(test_df[col])
                label_encoders[col] = le

            # Save preprocessing object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            joblib.dump(label_encoders, self.data_transformation_config.preprocessor_obj_file_path)
            logging.info(f"Preprocessing object saved at {self.data_transformation_config.preprocessor_obj_file_path}")

            # Save training columns (all feature columns, assume last column is target)
            # We will save columns order excluding the last column (target) because your trainer expects last col as target.
            train_columns = train_df.columns[:-1].tolist()  # ALL except last column
            joblib.dump(train_columns, self.data_transformation_config.train_columns_file_path)
            logging.info(f"Training columns saved at {self.data_transformation_config.train_columns_file_path}")

            return train_df, test_df, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error("Exception occurred in Data Transformation", exc_info=True)
            raise CustomException(e, sys)
