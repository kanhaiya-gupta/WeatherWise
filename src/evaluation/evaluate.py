import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

def evaluate_model(test_X_path, test_y_path, model_path, output_dir):
    """Evaluate the model and save metrics/plots."""
    X_test = pd.read_csv(test_X_path)
    y_test = pd.read_csv(test_y_path).values.ravel()
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
    }
    
    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Metrics saved to {output_dir}/metrics.json")
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model('../data/processed/test_X.csv', '../data/processed/test_y.csv', 
                   '../models/model.pkl', '../reports')
