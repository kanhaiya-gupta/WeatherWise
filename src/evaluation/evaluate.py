from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json

class WeatherModelEvaluator:
    """Class for evaluating weather prediction models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: Dictionary containing evaluation configuration
                - reports_dir: Directory to save evaluation reports
                - metrics_file: Path to save metrics
                - plots_dir: Directory to save plots
                - mlflow: Dictionary containing MLflow configuration
                    - tracking_uri: MLflow tracking URI
                    - experiment_name: MLflow experiment name
        """
        self.config = config
        self.reports_dir = Path(config['reports_dir'])
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            f"{prefix}accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}precision": precision_score(y_true, y_pred),
            f"{prefix}recall": recall_score(y_true, y_pred),
            f"{prefix}f1": f1_score(y_true, y_pred)
        }
        return metrics
        
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix"
    ) -> str:
        """
        Create and save confusion matrix plot.
        
        Returns:
            Path to saved plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(title)
        
        plot_path = self.reports_dir / "confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close()
        
        return str(plot_path)
        
    def plot_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Classification Report"
    ) -> str:
        """
        Create and save classification report plot.
        
        Returns:
            Path to saved plot
        """
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(report_df.iloc[:-3, :-1], annot=True, cmap='Blues')
        plt.title(title)
        
        plot_path = self.reports_dir / "classification_report.png"
        plt.savefig(plot_path)
        plt.close()
        
        return str(plot_path)
        
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        title: str = "Feature Importance"
    ) -> Optional[str]:
        """
        Create and save feature importance plot if the model supports it.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            title: Plot title
            
        Returns:
            Path to saved plot if successful, None otherwise
        """
        try:
            # Try to get feature importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                return None
                
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(title)
            
            plot_path = self.reports_dir / "feature_importance.png"
            plt.savefig(plot_path)
            plt.close()
            
            return str(plot_path)
            
        except Exception:
            return None
            
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: Any,
        feature_names: List[str],
        dataset_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance and log results with MLflow.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model: Trained model
            feature_names: List of feature names
            dataset_name: Name of the dataset (e.g., "test", "validation")
            
        Returns:
            Dictionary of metrics
        """
        with mlflow.start_run(nested=True):
            # Calculate metrics
            metrics = self.calculate_metrics(y_true, y_pred, prefix=f"{dataset_name}_")
            
            # Create and save plots
            cm_plot = self.plot_confusion_matrix(
                y_true, y_pred,
                title=f"Confusion Matrix ({dataset_name} set)"
            )
            report_plot = self.plot_classification_report(
                y_true, y_pred,
                title=f"Classification Report ({dataset_name} set)"
            )
            
            # Try to create feature importance plot
            feature_importance_plot = self.plot_feature_importance(
                model, feature_names,
                title=f"Feature Importance ({dataset_name} set)"
            )
            
            # Log metrics and artifacts with MLflow
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(cm_plot)
            mlflow.log_artifact(report_plot)
            if feature_importance_plot:
                mlflow.log_artifact(feature_importance_plot)
            
            # Save metrics to file
            metrics_file = self.reports_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            return metrics