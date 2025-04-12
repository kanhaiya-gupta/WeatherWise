import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from src.utils.utils_and_constants import BASE_DIR, REPORTS_DIR # type: ignore


def plot_confusion_matrix(model, X_test, y_test):
    plot_confusion_matrix_file = os.path.join(REPORTS_DIR, 'confusion_matrix.png')
    _ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
    plt.savefig(plot_confusion_matrix_file)


def save_metrics(metrics):
    # Define the full path for the metrics file
    metrics_file = os.path.join(REPORTS_DIR, 'metrics.json')

    with open(metrics_file, "w") as fp:
        json.dump(metrics, fp)
    print(f"Metrics saved to: {metrics_file}")
