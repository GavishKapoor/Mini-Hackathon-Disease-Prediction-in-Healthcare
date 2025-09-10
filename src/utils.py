# src/utils.py
import os
import joblib

def save_model(file_path: str, model):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
