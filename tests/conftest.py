import pytest
from fastapi.testclient import TestClient
import pandas as pd
import joblib
from pathlib import Path
import yaml
import sys
import os

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.app import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_weather_data():
    return {
        "Location": "Sydney",
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

@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def model(config):
    model_path = Path(config['model']['path'])
    if not model_path.exists():
        pytest.skip("Model file not found")
    return joblib.load(model_path) 