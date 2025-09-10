import os
import sys
import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

def run_training_pipeline():
    try:
        logging.info("Training pipeline started")

        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        print(f"Train and Test data saved at:\n{train_path}\n{test_path}")

        # Step 2: Data Transformation
        transformation = DataTransformation()
        train_df, test_df, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)
        print(f"Preprocessor saved at: {preprocessor_path}")

        # Step 3: Model Training
        trainer = ModelTrainer()
        accuracy = trainer.initiate_model_trainer(train_df.to_numpy(), test_df.to_numpy())
        print(f"Training completed! Model accuracy: {accuracy}")

        logging.info("Training pipeline completed successfully")

    except Exception as e:
        logging.error("Exception occurred in training pipeline", exc_info=True)
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
