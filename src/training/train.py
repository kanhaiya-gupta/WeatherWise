import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from pathlib import Path
from typing import Dict, Any, Tuple, List
import yaml

from src.utils.utils_and_constants import (
    PROCESSED_DATASET,
    TARGET_COLUMN,
    REPORTS,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME
)
from src.models.model import ModelFactory


class WeatherModelTrainer:
    """Class for training and evaluating weather prediction models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Dictionary containing training configuration
                - dataset: Dictionary containing dataset configuration
                    - drop_columns: List of columns to drop
                    - target_column: Name of target column
                    - paths: Dictionary of dataset paths
                - model: Dictionary containing model configuration
                    - type: Model type ('random_forest', 'neural_network', 'logistic')
                    - params_file: Path to JSON file containing model parameters
                - training: Dictionary containing training configuration
                    - test_size: Proportion of data to use for testing
                    - random_state: Random seed for reproducibility
                    - cv_folds: Number of cross-validation folds
                - mlflow: Dictionary containing MLflow configuration
                    - tracking_uri: MLflow tracking URI
                    - experiment_name: MLflow experiment name
        """
        self.config = config
        self.dataset_config = config['dataset']
        self.model_config = config['model']
        self.training_config = config['training']
        self.mlflow_config = config['mlflow']
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        mlflow.set_experiment(self.mlflow_config['experiment_name'])
        
        # Load model parameters from file if specified
        params = None
        if 'params_file' in self.model_config:
            params_file = Path(self.model_config['params_file'])
            if params_file.exists():
                try:
                    with open(params_file, 'r') as f:
                        params = json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load parameters from {params_file}: {e}")
        
        # Initialize model using factory
        self.model = ModelFactory.create_model(
            model_type=self.model_config['type'],
            params=params
        )
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load processed data and split into features and target."""
        data_path = Path(self.dataset_config['paths']['processed'])
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data file not found at {data_path}")
            
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            raise ValueError(f"Failed to load data from {data_path}: {e}")
        
        # Drop specified columns
        df = df.drop(columns=self.dataset_config['drop_columns'], errors='ignore')
        
        # Separate features and target
        target_column = self.dataset_config['target_column']
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Convert target to binary (0/1) if it's categorical
        if y.dtype == 'object':
            y = (y == 'Yes').astype(int)
            
        return X, y
        
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data into training and testing sets."""
        return train_test_split(
            X, y,
            test_size=self.training_config['test_size'],
            random_state=self.training_config['random_state'],
            stratify=y
        )
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
    def perform_cross_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform cross-validation and return mean scores."""
        cv_scores = cross_val_score(
            self.model.model, X, y,
            cv=self.training_config['cv_folds'],
            scoring='accuracy'
        )
        return {
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std()
        }
        
    def save_model(self, model_path: str = None) -> None:
        """Save the trained model to disk."""
        if model_path is None:
            model_path = self.model_config['path']
            
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(self.model.model, path)
        except Exception as e:
            raise ValueError(f"Failed to save model to {path}: {e}")
        
    def save_metrics(self, metrics: Dict[str, float]) -> None:
        """Save evaluation metrics to a JSON file."""
        reports_dir = self.config['directories']['reports']
        os.makedirs(reports_dir, exist_ok=True)
        metrics_path = os.path.join(reports_dir, 'metrics.json')
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to save metrics to {metrics_path}: {e}")
            
    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Plot and save confusion matrix."""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        reports_dir = self.config['directories']['reports']
        os.makedirs(reports_dir, exist_ok=True)
        try:
            plt.savefig(os.path.join(reports_dir, 'confusion_matrix.png'))
        except Exception as e:
            raise ValueError(f"Failed to save confusion matrix: {e}")
        finally:
            plt.close()
        
    def train(self, save_model: bool = True) -> Dict[str, float]:
        """
        Train the model and evaluate its performance.
        
        Args:
            save_model: Whether to save the trained model
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            with mlflow.start_run(nested=True):
                # Load and split data
                X, y = self.load_data()
                X_train, X_test, y_train, y_test = self.split_data(X, y)
                
                # Log model parameters
                mlflow.log_params({
                    'model_type': type(self.model.model).__name__,
                    'test_size': self.training_config['test_size'],
                    'random_state': self.training_config['random_state'],
                    'target_column': self.dataset_config['target_column'],
                    'drop_columns': self.dataset_config['drop_columns'],
                    'feature_columns': list(X.columns),
                    **self.model.get_params()  # Log all model parameters
                })
                
                # Train model
                self.model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = self.model.predict(X_test)
                
                # Evaluate model
                metrics = self.evaluate_model(y_test, y_pred)
                cv_metrics = self.perform_cross_validation(X, y)
                
                # Log metrics
                mlflow.log_metrics({**metrics, **cv_metrics})
                
                # Log model
                mlflow.sklearn.log_model(self.model.model, "model")
                
                if save_model:
                    self.save_model()
                    # Log the model as an artifact
                    model_path = self.model_config['path']
                    mlflow.log_artifact(str(Path(model_path)))
                    
                # Save metrics and plots
                self.save_metrics({**metrics, **cv_metrics})
                self.plot_confusion_matrix(X_test, y_test)
                
                return {**metrics, **cv_metrics}
                
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")


def main():
    """Main function to run training pipeline."""
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # Train model
        trainer = WeatherModelTrainer(config)
        metrics = trainer.train()
        
        print("Model training completed with metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()