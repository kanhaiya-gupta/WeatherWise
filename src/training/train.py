import pandas as pd
import joblib
from src.models.model import define_model
import os

def train_model(train_X_path, train_y_path, model_path):
    """Train the model and save it."""
    X_train = pd.read_csv(train_X_path)
    y_train = pd.read_csv(train_y_path).values.ravel()
    
    model = define_model()
    model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model('../data/processed/train_X.csv', '../data/processed/train_y.csv', '../models/model.pkl')
