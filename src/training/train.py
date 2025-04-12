import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.metrics_and_plots import plot_confusion_matrix, save_metrics
from src.models.model import train_model
from src.evaluation.evaluate import evaluate_model
from src.utils.utils_and_constants import PROCESSED_DATASET, TARGET_COLUMN, MODEL_DIR


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y


def save_model(model, model_type, save_dir="models"):
    """
    Save the trained model as a .pkl file in the specified directory.

    Parameters:
    - model: Trained model to save
    - model_type: String specifying the model type for filename
    - save_dir: Directory to save the model (default: 'models')
    """
    # Create the directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    filename = os.path.join(save_dir, f"trained_model_{model_type}.pkl")
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")


def main(model_type="random_forest"):
    X, y = load_data(PROCESSED_DATASET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)

    model = train_model(X_train, y_train, model_type=model_type)
    metrics = evaluate_model(model, X_test, y_test)

    print(f"====================Test Set Metrics ({model_type})==================")
    print(json.dumps(metrics, indent=2))
    print("======================================================")

    save_metrics(metrics)
    plot_confusion_matrix(model, X_test, y_test)
    save_model(model, model_type)


if __name__ == "__main__":
    # Try different models
    for model_type in ["random_forest", "neural_network", "logistic"]:
        main(model_type)
