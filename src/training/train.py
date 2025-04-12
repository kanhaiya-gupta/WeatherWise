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
    for model_type in ["random_forest"]:
        main(model_type)
    #for model_type in ["random_forest", "neural_network", "logistic"]:
    #    main(model_type)
    

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

from scripts.metrics_and_plots import plot_confusion_matrix, save_metrics
from src.models.model import train_model
from src.evaluation.evaluate import evaluate_model
from src.utils.utils_and_constants import PROCESSED_DATASET, TARGET_COLUMN, MODEL_DIR, REPORTS


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
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    filename = os.path.join(save_dir, f"trained_model_{model_type}.pkl")
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")


def main(model_type="random_forest"):
    # Load and split data
    X, y = load_data(PROCESSED_DATASET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)

    # Train model
    model = train_model(X_train, y_train, model_type=model_type)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Generate and save predictions for confusion matrix
    y_pred = model.predict(X_test)
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred
    })
    
    predictions_df.to_csv(f'{REPORTS}/predictions.csv', index=False)
    print("Predictions saved to reports/predictions.csv")

    # Generate and save ROC curve data
    # Use predict_proba if available, else fall back to decision_function
    try:
        y_scores = model.predict_proba(X_test)[:, 1]  # Probability for positive class
    except AttributeError:
        y_scores = model.decision_function(X_test)  # For models like LogisticRegression
    
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_df = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr
    })
    roc_df.to_csv(f'{REPORTS}/roc_curve.csv', index=False)
    print("ROC curve data saved to reports/roc_curve.csv")

    # Print and save metrics
    print(f"====================Test Set Metrics ({model_type})==================")
    print(json.dumps(metrics, indent=2))
    print("======================================================")
    save_metrics(metrics)
    
    # Generate confusion matrix plot
    plot_confusion_matrix(model, X_test, y_test)
    
    # Save model
    save_model(model, model_type)


if __name__ == "__main__":
    # Train random_forest model (extend to others if needed)
    for model_type in ["random_forest"]:
        main(model_type)
    # Uncomment to train multiple models
    # for model_type in ["random_forest", "neural_network", "logistic"]:
    #     main(model_type)