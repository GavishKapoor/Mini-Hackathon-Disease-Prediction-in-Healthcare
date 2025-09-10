import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_train_data_path: str = os.path.join('artifacts', 'raw_train.csv')
    raw_test_data_path: str = os.path.join('artifacts', 'raw_test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            train_df = pd.read_csv("notebooks/data/training.csv")
            test_df = pd.read_csv("notebooks/data/testing.csv")
            logging.info("Training and testing datasets read successfully")

            # Ensure artifacts folder exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw files
            train_df.to_csv(self.ingestion_config.raw_train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.raw_test_data_path, index=False, header=True)
            logging.info("Raw train and test data saved")

            # Save processed copies
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data copied to artifacts folder")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error("Exception occurred at Data Ingestion stage", exc_info=True)
            raise CustomException(e, sys)


