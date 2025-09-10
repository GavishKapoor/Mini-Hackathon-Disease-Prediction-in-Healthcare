from src.components.data_ingestion import DataIngestion
from src.logger import logging

if __name__ == "__main__":
    try:
        logging.info("Training Pipeline Test Start")

        # Step 1: Run Data Ingestion only
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        logging.info(f"Train CSV path: {train_path}")
        logging.info(f"Test CSV path: {test_path}")

        print(f"Train CSV: {train_path}")
        print(f"Test CSV: {test_path}")
        print("Data Ingestion in training pipeline ran successfully!")

    except Exception as e:
        logging.info("Exception occurred in training pipeline")
        raise e
