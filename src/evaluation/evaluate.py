import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(model, X_test, y_test, float_precision=4):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return json.loads(
        json.dumps(metrics), parse_float=lambda x: round(float(x), float_precision)
    )