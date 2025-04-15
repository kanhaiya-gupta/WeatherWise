import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys
import os

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.app import app, preprocess_input, preprocessor  # Import preprocessor from app
from src.preprocessing.preprocess import WeatherDataPreprocessor

# Create test client
client = TestClient(app)

# Load config from project root
config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Sample test data
SAMPLE_WEATHER_DATA = {
    "Location": "Albury",  # Using a location that exists in training data
    "MinTemp": 13.4,
    "MaxTemp": 22.9,
    "Rainfall": 0.6,
    "Evaporation": 4.5,
    "Sunshine": 6.7,
    "WindGustDir": "NNW",
    "WindGustSpeed": 44.0,
    "WindDir9am": "NW",
    "WindDir3pm": "NNW",
    "WindSpeed9am": 20.0,
    "WindSpeed3pm": 24.0,
    "Humidity9am": 71.0,
    "Humidity3pm": 22.0,
    "Pressure9am": 1007.7,
    "Pressure3pm": 1007.1,
    "Cloud9am": 8.0,
    "Cloud3pm": 5.0,
    "Temp9am": 16.9,
    "Temp3pm": 21.8,
    "RainToday": "No",
    "RISK_MM": 0.0
}

# Initialize preprocessor for tests
preprocessor = WeatherDataPreprocessor(config)

# Create training data with multiple samples to better fit the preprocessor
training_data = pd.DataFrame([
    SAMPLE_WEATHER_DATA,
    {**SAMPLE_WEATHER_DATA, 'Location': 'Melbourne', 'RainToday': 'Yes'},
    {**SAMPLE_WEATHER_DATA, 'Location': 'Brisbane', 'RainToday': 'No'},
    {**SAMPLE_WEATHER_DATA, 'Location': 'Perth', 'RainToday': 'Yes'}
])
training_data['RainTomorrow'] = ['No', 'Yes', 'No', 'Yes']  # Add target column
X, y = preprocessor.process(data=training_data, save_processed=False)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "mlflow_connected" in data
    assert "config_loaded" in data
    assert "preprocessor_fitted" in data

def test_model_info():
    """Test the model info endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "model_path" in data
    assert "mlflow_tracking_uri" in data
    assert "feature_columns" in data
    assert "categorical_columns" in data

def test_predict_endpoint():
    """Test the prediction endpoint with valid data."""
    response = client.post("/predict", json=SAMPLE_WEATHER_DATA)
    assert response.status_code == 200
    data = response.json()
    assert "rain_probability" in data
    assert "rain_tomorrow" in data
    assert isinstance(data["rain_probability"], float)
    assert data["rain_tomorrow"] in ["Yes", "No"]

def test_predict_endpoint_invalid_data():
    """Test the prediction endpoint with invalid data."""
    invalid_data = {
        "Location": "Sydney",
        "MinTemp": "invalid",  # Invalid type
        "MaxTemp": 22.9,
        # Missing other required fields
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_preprocess_input():
    """Test the preprocessing of input data."""
    # Convert sample data to DataFrame
    input_data = pd.DataFrame([SAMPLE_WEATHER_DATA])
    
    # Preprocess the data
    processed_data = preprocess_input(input_data)
    
    # Check that the output is a DataFrame
    assert isinstance(processed_data, pd.DataFrame)
    
    # Check that there are no missing values
    assert not processed_data.isnull().any().any()
    
    # Check that all features are numeric
    for col in processed_data.columns:
        assert pd.api.types.is_numeric_dtype(processed_data[col])

    # Check that all required features are present
    required_features = preprocessor.feature_columns
    assert all(feature in processed_data.columns for feature in required_features)

    # Check that categorical features are properly encoded
    categorical_features = preprocessor.categorical_columns
    for col in categorical_features:
        assert pd.api.types.is_numeric_dtype(processed_data[col])
        assert not processed_data[col].isnull().any()

    # Check that numeric features are finite (no inf or nan)
    numeric_features = [col for col in required_features if col not in categorical_features]
    for col in numeric_features:
        assert np.all(np.isfinite(processed_data[col]))

def test_preprocess_input_edge_cases():
    """Test preprocessing with edge cases."""
    # Test with boundary values
    edge_data = SAMPLE_WEATHER_DATA.copy()
    edge_data.update({
        'Location': 'Albury',  # Using a location that exists in training data
        'MinTemp': -10.0,  # Very low temperature
        'MaxTemp': 50.0,   # Very high temperature
        'Rainfall': 500.0, # Heavy rainfall
        'WindGustSpeed': 200.0,  # Extreme wind
        'Humidity9am': 100.0,    # Maximum humidity
        'Humidity3pm': 0.0       # Minimum humidity
    })
    
    processed_edge = preprocess_input(pd.DataFrame([edge_data]))
    assert isinstance(processed_edge, pd.DataFrame)
    assert not processed_edge.isnull().any().any()
    assert np.all(np.isfinite(processed_edge))  # Check for no inf values
    
    # Test with zero values
    zero_data = SAMPLE_WEATHER_DATA.copy()
    zero_data.update({
        'Rainfall': 0.0,
        'Evaporation': 0.0,
        'WindGustSpeed': 0.0,
        'WindSpeed9am': 0.0,
        'WindSpeed3pm': 0.0
    })
    
    processed_zero = preprocess_input(pd.DataFrame([zero_data]))
    assert isinstance(processed_zero, pd.DataFrame)
    assert not processed_zero.isnull().any().any()
    assert np.all(np.isfinite(processed_zero))  # Check for no inf values
    
    # Test with known categorical values
    unusual_data = SAMPLE_WEATHER_DATA.copy()
    unusual_data.update({
        'Location': 'Albury',  # Using a location that exists in training data
        'WindGustDir': 'NNW',  # Using a known direction from training data
        'RainToday': 'No'      # Using a known value
    })
    
    processed_unusual = preprocess_input(pd.DataFrame([unusual_data]))
    assert isinstance(processed_unusual, pd.DataFrame)
    assert not processed_unusual.isnull().any().any()
    assert np.all(np.isfinite(processed_unusual))  # Check for no inf values

def test_preprocess_input_invalid_data():
    """Test preprocessing with invalid data."""
    # Create invalid data
    invalid_data = pd.DataFrame({
        'Location': ['InvalidLocation'],
        'MinTemp': ['not_a_number']
    })
    
    # Test that preprocessing raises appropriate error
    with pytest.raises(Exception):
        preprocess_input(invalid_data)

def test_preprocess_input_missing_columns():
    """Test preprocessing with missing columns."""
    # Create data with missing columns
    incomplete_data = pd.DataFrame({
        'Location': [SAMPLE_WEATHER_DATA['Location']],
        'MinTemp': [SAMPLE_WEATHER_DATA['MinTemp']]
        # Missing other required columns
    })
    
    # Test that preprocessing raises appropriate error
    with pytest.raises(ValueError) as exc_info:
        preprocess_input(incomplete_data)
    assert "Missing required features" in str(exc_info.value) 