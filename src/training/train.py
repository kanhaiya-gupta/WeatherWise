# import pandas as pd
# import joblib
# from src.models.model import define_model
# import os

# def train_model(train_X_path, train_y_path, model_path):
#     """Train the model and save it."""
#     X_train = pd.read_csv(train_X_path)
#     y_train = pd.read_csv(train_y_path).values.ravel()
    
#     model = define_model()
#     model.fit(X_train, y_train)
    
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}")

# if __name__ == "__main__":
#     train_model('../data/processed/train_X.csv', '../data/processed/train_y.csv', '../models/model.pkl')


import json
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

from scripts.metrics_and_plots import plot_confusion_matrix, save_metrics # type: ignore
from src.models.model import evaluate_model, train_model # type: ignore
from src.utils.utils_and_constants import PROCESSED_DATASET, TARGET_COLUMN # type: ignore


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y


def main():
    X, y = load_data(PROCESSED_DATASET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    print("====================Test Set Metrics==================")
    print(json.dumps(metrics, indent=2))
    print("======================================================")

    save_metrics(metrics)
    plot_confusion_matrix(model, X_test, y_test)


if __name__ == "__main__":
    main()
