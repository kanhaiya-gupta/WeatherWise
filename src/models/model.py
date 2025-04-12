from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

RFC_FOREST_DEPTH = 2


def train_model(X_train, y_train, model_type="random_forest"):
    """
    Train a model based on the specified model_type.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - model_type: String specifying the model ('random_forest', 'neural_network', 'logistic')

    Returns:
    - Trained model
    """
    if model_type == "random_forest":
        model = RandomForestClassifier(
            max_depth=RFC_FOREST_DEPTH, n_estimators=5, random_state=1993
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