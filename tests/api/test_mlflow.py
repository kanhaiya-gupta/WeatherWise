import pytest
import mlflow
import os
from pathlib import Path
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def config():
    """Load configuration from config.yaml."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def mlflow_tracking_uri(tmp_path):
    """Set up a temporary MLflow tracking URI."""
    tracking_uri = str(tmp_path / "mlruns")
    os.makedirs(tracking_uri, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri

@pytest.fixture(scope="session")
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

def test_mlflow_experiment_creation(config, mlflow_tracking_uri):
    """Test creating an MLflow experiment."""
    experiment_name = "test_experiment"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    assert experiment_id is not None
    
    experiment = mlflow.get_experiment(experiment_id)
    assert experiment.name == experiment_name

def test_mlflow_run_logging(config, mlflow_tracking_uri, sample_training_data):
    """Test logging metrics and parameters to MLflow."""
    from src.preprocessing.preprocess import WeatherDataPreprocessor
    
    # Set up experiment
    mlflow.set_experiment("test_logging")
    
    with mlflow.start_run() as run:
        # Process data
        preprocessor = WeatherDataPreprocessor(config)
        X = sample_training_data.drop('RainTomorrow', axis=1)
        y = sample_training_data['RainTomorrow']
        X_processed = preprocessor.process(X)
        
        # Train model with some parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        model = RandomForestClassifier(**params)
        model.fit(X_processed, y)
        
        # Log parameters
        mlflow.log_params(params)
        
        # Make predictions and calculate metrics
        y_pred = model.predict(X_processed)
        metrics = {
            'accuracy': (y == y_pred).mean(),
            'feature_count': X_processed.shape[1]
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Get the run data
        run_data = mlflow.get_run(run.info.run_id)
        
        # Verify logged data
        assert run_data.data.params == {str(k): str(v) for k, v in params.items()}
        assert all(
            run_data.data.metrics[k] == v 
            for k, v in metrics.items()
        )

def test_mlflow_model_loading(config, mlflow_tracking_uri, sample_training_data):
    """Test loading a model from MLflow."""
    from src.preprocessing.preprocess import WeatherDataPreprocessor
    
    # Set up experiment and log a model
    mlflow.set_experiment("test_model_loading")
    
    with mlflow.start_run() as run:
        # Process data
        preprocessor = WeatherDataPreprocessor(config)
        X = sample_training_data.drop('RainTomorrow', axis=1)
        y = sample_training_data['RainTomorrow']
        X_processed = preprocessor.process(X)
        
        # Train and log model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_processed, y)
        mlflow.sklearn.log_model(model, "model")
        
        # Get run ID
        run_id = run.info.run_id
    
    # Load model from MLflow
    model_uri = f"runs:/{run_id}/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    
    # Compare predictions
    original_preds = model.predict(X_processed)
    loaded_preds = loaded_model.predict(X_processed)
    
    assert np.array_equal(original_preds, loaded_preds)

def test_mlflow_artifact_logging(config, mlflow_tracking_uri, tmp_path):
    """Test logging and retrieving artifacts in MLflow."""
    # Create a test artifact
    artifact_path = tmp_path / "feature_importance.csv"
    feature_importance = pd.DataFrame({
        'feature': ['feature1', 'feature2'],
        'importance': [0.7, 0.3]
    })
    feature_importance.to_csv(artifact_path, index=False)
    
    # Log artifact
    mlflow.set_experiment("test_artifacts")
    with mlflow.start_run() as run:
        mlflow.log_artifact(str(artifact_path))
        
        # Get run ID
        run_id = run.info.run_id
    
    # Get artifact
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    
    assert len(artifacts) == 1
    assert artifacts[0].path == "feature_importance.csv" 