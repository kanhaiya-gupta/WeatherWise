import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from src.utils.utils_and_constants import BEST_PAR_PATH


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the model.
        
        Args:
            params: Dictionary of model parameters
        """
        self.params = params or {}
        self.model: Optional[BaseEstimator] = None
        
    @abstractmethod
    def create_model(self) -> BaseEstimator:
        """Create and return the model instance."""
        pass
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the training data."""
        if self.model is None:
            self.model = self.create_model()
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)
        
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is None:
            return self.params
        return self.model.get_params()


class RandomForestModel(BaseModel):
    """Random Forest model implementation."""
    
    def create_model(self) -> BaseEstimator:
        """Create and return a RandomForestClassifier instance."""
        default_params = {
            'max_depth': 2,
            'n_estimators': 5,
            'random_state': 1993
        }
        params = {**default_params, **self.params}  # Merge defaults with provided params
        return RandomForestClassifier(**params)


class NeuralNetworkModel(BaseModel):
    """Neural Network model implementation."""
    
    def create_model(self) -> BaseEstimator:
        """Create and return an MLPClassifier instance."""
        default_params = {
            'hidden_layer_sizes': (100,),
            'max_iter': 200,
            'random_state': 1993
        }
        params = {**default_params, **self.params}
        return MLPClassifier(**params)


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model implementation."""
    
    def create_model(self) -> BaseEstimator:
        """Create and return a LogisticRegression instance."""
        default_params = {
            'random_state': 1993
        }
        params = {**default_params, **self.params}
        return LogisticRegression(**params)


class ModelFactory:
    """Factory class for creating model instances."""
    
    @staticmethod
    def create_model(model_type: str, params: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        Create a model instance based on the specified type.
        
        Args:
            model_type: Type of model to create (e.g., 'RandomForestClassifier', 'MLPClassifier', 'LogisticRegression')
            params: Dictionary of model parameters
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        model_type = model_type.strip()
        model_map = {
            'RandomForestClassifier': RandomForestModel,
            'MLPClassifier': NeuralNetworkModel,
            'LogisticRegression': LogisticRegressionModel
        }
        
        model_class = model_map.get(model_type)
        if model_class is None:
            raise ValueError(f"Unsupported model_type: {model_type}. Supported types: {list(model_map.keys())}")
        
        return model_class(params=params)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "RandomForestClassifier",
    params: Optional[Dict[str, Any]] = None
) -> BaseEstimator:
    """
    Train a model based on the specified model_type.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train
        params: Dictionary of model parameters
        
    Returns:
        Trained model
    """
    model = ModelFactory.create_model(model_type, params)
    model.fit(X_train, y_train)
    return model.model