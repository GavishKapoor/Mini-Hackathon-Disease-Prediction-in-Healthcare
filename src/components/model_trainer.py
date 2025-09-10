import os
import sys
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from dataclasses import dataclass
from src.utils import save_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            rf_model = RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight="balanced"
            )

            logging.info("Training RandomForestClassifier")
            rf_model.fit(X_train, y_train)

            y_pred = rf_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            print(f"Accuracy: {accuracy}")
            print(f"Classification Report:\n{class_report}")
            print(f"Confusion Matrix:\n{conf_matrix}")

            # Save model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_model(self.model_trainer_config.trained_model_file_path, rf_model)
            logging.info(f"Trained model saved at {self.model_trainer_config.trained_model_file_path}")

            return accuracy

        except Exception as e:
            logging.error("Exception occurred in ModelTrainer", exc_info=True)
            raise e

