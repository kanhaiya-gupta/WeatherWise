import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

@pytest.fixture
def sample_training_data():
    """Create sample data for model training."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Location': np.random.choice(['Sydney', 'Melbourne', 'Brisbane'], n_samples),
        'MinTemp': np.random.normal(15, 5, n_samples),
        'MaxTemp': np.random.normal(25, 5, n_samples),
        'Rainfall': np.random.exponential(5, n_samples),
        'Evaporation': np.random.normal(5, 2, n_samples),
        'Sunshine': np.random.normal(8, 3, n_samples),
        'WindGustDir': np.random.choice(['N', 'S', 'E', 'W'], n_samples),
        'WindGustSpeed': np.random.normal(30, 10, n_samples),
        'WindDir9am': np.random.choice(['N', 'S', 'E', 'W'], n_samples),
        'WindDir3pm': np.random.choice(['N', 'S', 'E', 'W'], n_samples),
        'WindSpeed9am': np.random.normal(15, 5, n_samples),
        'WindSpeed3pm': np.random.normal(20, 5, n_samples),
        'Humidity9am': np.random.normal(70, 10, n_samples),
        'Humidity3pm': np.random.normal(50, 15, n_samples),
        'Pressure9am': np.random.normal(1015, 5, n_samples),
        'Pressure3pm': np.random.normal(1013, 5, n_samples),
        'Cloud9am': np.random.randint(0, 9, n_samples),
        'Cloud3pm': np.random.randint(0, 9, n_samples),
        'Temp9am': np.random.normal(18, 5, n_samples),
        'Temp3pm': np.random.normal(23, 5, n_samples),
        'RainToday': np.random.choice(['Yes', 'No'], n_samples),
        'RISK_MM': np.random.exponential(5, n_samples),
        'RainTomorrow': np.random.choice(['Yes', 'No'], n_samples)
    }
    return pd.DataFrame(data)

@pytest.fixture
def config():
    """Load configuration from config.yaml."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_model_training(sample_training_data, config):
    """Test model training process."""
    from src.preprocessing.preprocess import WeatherDataPreprocessor
    
    # Initialize preprocessor
    preprocessor = WeatherDataPreprocessor(config)
    
    # Process training data
    X_train = sample_training_data.drop('RainTomorrow', axis=1)
    y_train = sample_training_data['RainTomorrow']
    
    # Fit preprocessor on training data
    preprocessor.fit(X_train)
    
    # Process data
    X_processed = preprocessor.transform_new_data(X_train)
    
    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_processed, y_train)
    
    # Make predictions
    y_pred = model.predict(X_processed)
    
    # Check model performance
    assert accuracy_score(y_train, y_pred) > 0.5  # Basic performance check
    assert len(model.feature_names_in_) == len(config['dataset']['feature_columns'])

def test_model_evaluation(sample_training_data, config):
    """Test model evaluation metrics."""
    from src.preprocessing.preprocess import WeatherDataPreprocessor
    
    # Initialize preprocessor and process data
    preprocessor = WeatherDataPreprocessor(config)
    X = sample_training_data.drop('RainTomorrow', axis=1)
    y = sample_training_data['RainTomorrow']
    
    # Fit and transform data
    preprocessor.fit(X)
    X_processed = preprocessor.transform_new_data(X)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_processed, y)
    
    # Make predictions
    y_pred = model.predict(X_processed)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, pos_label='Yes'),
        'recall': recall_score(y, y_pred, pos_label='Yes'),
        'f1': f1_score(y, y_pred, pos_label='Yes')
    }
    
    # Check metrics are within reasonable ranges
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1

def test_model_save_load(sample_training_data, config, tmp_path):
    """Test model saving and loading functionality."""
    from src.preprocessing.preprocess import WeatherDataPreprocessor
    
    # Initialize preprocessor and process data
    preprocessor = WeatherDataPreprocessor(config)
    X = sample_training_data.drop('RainTomorrow', axis=1)
    y = sample_training_data['RainTomorrow']
    
    # Fit and transform data
    preprocessor.fit(X)
    X_processed = preprocessor.transform_new_data(X)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_processed, y)
    
    # Save model
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)
    
    # Load model
    loaded_model = joblib.load(model_path)
    
    # Compare predictions
    original_preds = model.predict(X_processed)
    loaded_preds = loaded_model.predict(X_processed)
    
    # Check predictions match
    assert np.array_equal(original_preds, loaded_preds)
    assert hasattr(loaded_model, 'feature_names_in_')

def test_model_feature_importance(sample_training_data, config):
    """Test feature importance analysis."""
    from src.preprocessing.preprocess import WeatherDataPreprocessor
    
    # Initialize preprocessor and process data
    preprocessor = WeatherDataPreprocessor(config)
    X = sample_training_data.drop('RainTomorrow', axis=1)
    y = sample_training_data['RainTomorrow']
    
    # Fit and transform data
    preprocessor.fit(X)
    X_processed = preprocessor.transform_new_data(X)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_processed, y)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Check feature importance properties
    assert len(importance) == X_processed.shape[1]
    assert np.all(importance >= 0)
    assert np.isclose(np.sum(importance), 1.0)
    
    # Check top features have reasonable importance
    top_importance = np.max(importance)
    assert top_importance > 0.05  # At least one feature should be somewhat important 