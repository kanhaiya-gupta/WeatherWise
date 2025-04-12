import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from src.utils.utils_and_constants import BEST_PAR_PATH

def train_model(X_train, y_train, model_type="random_forest"):
    """
    Train a model based on the specified model_type, using best parameters from BEST_PAR_PATH for random forest.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - model_type: String specifying the model ('random_forest', 'neural_network', 'logistic')

    Returns:
    - Trained model
    """
    if model_type == "random_forest":
        # Load best parameters from JSON file
        try:
            with open(BEST_PAR_PATH, 'r') as f:
                best_params = json.load(f)
            model = RandomForestClassifier(
                max_depth=best_params.get("max_depth", 2),
                n_estimators=best_params.get("n_estimators", 5),
                random_state=best_params.get("random_state", 1993)
            )
        except FileNotFoundError:
            # Fallback to default parameters if file not found
            model = RandomForestClassifier(
                max_depth=2, n_estimators=5, random_state=1993
            )
    elif model_type == "neural_network":
        model = MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=200, random_state=1993
        )
    elif model_type == "logistic":
        model = LogisticRegression(random_state=1993)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train, y_train)
    return model