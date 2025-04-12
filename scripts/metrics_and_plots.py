import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# def generate_metrics_plot(metrics_path, output_path):
#     """Generate a bar plot of metrics."""
#     with open(metrics_path, 'r') as f:
#         metrics = json.load(f)
    
#     plt.figure(figsize=(8, 5))
#     plt.bar(metrics.keys(), metrics.values(), color='skyblue')
#     plt.title('Model Performance Metrics')
#     plt.ylabel('Score')
#     plt.ylim(0, 1)
#     plt.savefig(output_path)
#     plt.close()
#     print(f"Metrics plot saved to {output_path}")

# if __name__ == "__main__":
#     generate_metrics_plot('../reports/metrics.json', '../reports/metrics_bar.png')

# Helper function to ensure reports directory exists
def ensure_reports_directory():
    # Get the base directory (root of your project)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Define the path for the 'reports' folder
    reports_dir = os.path.join(base_dir, 'reports')
    
    # Create the 'reports' directory if it doesn't exist
    os.makedirs(reports_dir, exist_ok=True)
    
    return reports_dir


def plot_confusion_matrix(model, X_test, y_test):
    # Ensure reports directory exists
    reports_dir = ensure_reports_directory()
    # Define the full path for the metrics file
    plot_confusion_matrix_file = os.path.join(reports_dir, 'confusion_matrix.png')

    _ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
    plt.savefig(plot_confusion_matrix_file)


def save_metrics(metrics):
    # Ensure reports directory exists
    reports_dir = ensure_reports_directory()
    # Define the full path for the metrics file
    metrics_file = os.path.join(reports_dir, 'metrics.json')

    with open(metrics_file, "w") as fp:
        json.dump(metrics, fp)
    print(f"Metrics saved to: {metrics_file}")
