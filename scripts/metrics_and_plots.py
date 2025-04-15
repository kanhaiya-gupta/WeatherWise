from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
import json
import mlflow

class MetricsAndPlots:
    """Class for generating and saving metrics and plots."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Args:
            config: Dictionary containing configuration
                - reports_dir: Directory to save reports
                - plots_dir: Directory to save plots
                - metrics_file: Path to save metrics
        """
        self.config = config
        self.reports_dir = Path(config['reports_dir'])
        self.plots_dir = Path(config['plots_dir'])
        
        # Create directories if they don't exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list = None,
        title: str = "Confusion Matrix"
    ) -> str:
        """
        Create and save confusion matrix plot.
        
        Returns:
            Path to saved plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plot_path = self.plots_dir / "confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close()
        
        return str(plot_path)
        
    def save_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Save metrics to JSON file.
        
        Returns:
            Path to saved metrics file
        """
        metrics_path = self.reports_dir / self.config['metrics_file']
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return str(metrics_path)
        
    def log_results(
        self,
        metrics: Dict[str, float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        run_name: str = "evaluation"
    ) -> None:
        """Log metrics and plots to MLflow."""
        with mlflow.start_run(run_name=run_name):
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Create and log confusion matrix
            plot_path = self.plot_confusion_matrix(y_true, y_pred)
            mlflow.log_artifact(plot_path)
            
            # Save and log metrics file
            metrics_path = self.save_metrics(metrics)
            mlflow.log_artifact(metrics_path)
