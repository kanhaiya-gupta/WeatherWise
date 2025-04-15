from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import mlflow
import joblib
import pandas as pd
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Union
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import datetime

from src.preprocessing.preprocess import WeatherDataPreprocessor
from src.utils.utils_and_constants import MODEL_PATH

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger('weatherwise')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        log_dir / f'weatherwise_{current_time}.log',
        encoding='utf-8'
    )
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define categorical columns
CATEGORICAL_COLUMNS = [
    'Location',
    'WindGustDir',
    'WindDir9am',
    'WindDir3pm',
    'RainToday'
]

app = FastAPI(
    title="WeatherWise API",
    description="API for predicting rainfall probability based on weather features",
    version="1.0.0"
)

# Initialize preprocessing components
try:
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    # Load model using joblib
    model_path = Path(config['model']['path'])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Successfully loaded model from {model_path}")
    
    # Initialize preprocessor
    preprocessor = WeatherDataPreprocessor(config)
    
    # Load and process training data
    if 'dataset' in config and 'paths' in config['dataset'] and 'raw' in config['dataset']['paths']:
        data_path = Path(config['dataset']['paths']['raw'])
        if data_path.exists():
            train_data = pd.read_csv(data_path)
            # Process the data to fit the preprocessor
            X, y = preprocessor.process(data=train_data, save_processed=False)
            logger.info("Successfully initialized preprocessor with training data")
        else:
            raise FileNotFoundError(f"Training data not found at {data_path}")
    else:
        raise ValueError("No data paths configured in config.yaml")
        
except Exception as e:
    logger.error(f"Error during initialization: {e}")
    raise

# Define input schema with constraints
class WeatherInput(BaseModel):
    Location: str = Field(..., description="Location name")
    MinTemp: float = Field(..., description="Minimum temperature")
    MaxTemp: float = Field(..., description="Maximum temperature")
    Rainfall: float = Field(..., description="Amount of rainfall")
    Evaporation: float = Field(..., description="Amount of evaporation")
    Sunshine: float = Field(..., description="Hours of sunshine")
    WindGustDir: str = Field(..., description="Direction of strongest wind gust")
    WindGustSpeed: float = Field(..., description="Speed of strongest wind gust")
    WindDir9am: str = Field(..., description="Wind direction at 9am")
    WindDir3pm: str = Field(..., description="Wind direction at 3pm")
    WindSpeed9am: float = Field(..., description="Wind speed at 9am")
    WindSpeed3pm: float = Field(..., description="Wind speed at 3pm")
    Humidity9am: float = Field(..., description="Humidity at 9am")
    Humidity3pm: float = Field(..., description="Humidity at 3pm")
    Pressure9am: float = Field(..., description="Pressure at 9am")
    Pressure3pm: float = Field(..., description="Pressure at 3pm")
    Cloud9am: float = Field(..., description="Cloud cover at 9am")
    Cloud3pm: float = Field(..., description="Cloud cover at 3pm")
    Temp9am: float = Field(..., description="Temperature at 9am")
    Temp3pm: float = Field(..., description="Temperature at 3pm")
    RainToday: str = Field(..., description="Whether it rained today (Yes/No)")
    RISK_MM: float = Field(..., description="Amount of rainfall risk")

    model_config = ConfigDict(json_schema_extra={
        "example": {
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
    })

def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess input data using the WeatherDataPreprocessor."""
    try:
        # Ensure all required features are present
        required_features = model.feature_names_in_
        missing_features = set(required_features) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Use the preprocessor to transform the data
        processed_data = preprocessor.transform_new_data(data)
        
        # Ensure columns match model's expected features
        processed_data = processed_data[required_features]
        
        return processed_data
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

class PredictionResponse(BaseModel):
    rain_probability: float = Field(..., description="Probability of rain tomorrow")
    rain_tomorrow: str = Field(..., description="Prediction of whether it will rain tomorrow (Yes/No)")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "rain_probability": 0.75,
            "rain_tomorrow": "Yes"
        }
    })

@app.post("/predict", response_model=PredictionResponse)
async def predict_rainfall(data: WeatherInput) -> Dict[str, Union[float, str]]:
    """
    Predict rainfall probability based on input weather features.
    
    Args:
        data: Weather features input
        
    Returns:
        Dictionary containing rain probability and prediction
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Log input data
        logger.info(f"Received prediction request with data: {data.model_dump()}")
        
        # Convert input to DataFrame
        input_data = pd.DataFrame([data.model_dump()])
        
        # Preprocess input data
        processed_data = preprocess_input(input_data)
        logger.info("Successfully preprocessed input data")
        
        # Make prediction
        prediction = model.predict_proba(processed_data)[:, 1][0]  # Probability of rain
        logger.info(f"Made prediction with probability: {prediction}")
        
        response = {
            "rain_probability": round(float(prediction), 4),
            "rain_tomorrow": "Yes" if prediction > 0.5 else "No"
        }
        logger.info(f"Returning response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Check API health status.
    
    Returns:
        Dictionary containing health status information
        
    Raises:
        HTTPException: If health check fails
    """
    try:
        # Check MLflow connection
        mlflow.get_tracking_uri()
        
        status = {
            "status": "healthy",
            "model_loaded": bool(model),
            "mlflow_connected": True,
            "config_loaded": bool(config),
            "preprocessor_fitted": hasattr(preprocessor, 'transform_new_data'),
            "model_loaded": bool(model)
        }
        logger.info(f"Health check completed: {status}")
        return status
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Health check error: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.
    
    Returns:
        Dictionary containing model information
        
    Raises:
        HTTPException: If retrieving model info fails
    """
    try:
        model_info = {
            "model_type": config['model']['type'],
            "model_path": str(config['model']['path']),
            "mlflow_tracking_uri": config['mlflow']['tracking_uri'],
            "feature_columns": list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else None,
            "categorical_columns": CATEGORICAL_COLUMNS
        }
        logger.info(f"Retrieved model information: {model_info}")
        return model_info
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )

@app.get("/features")
async def get_features():
    """Get list of required features for prediction."""
    try:
        features = {
            "feature_columns": config['dataset']['feature_columns'],
            "categorical_columns": config['dataset']['categorical_columns'],
            "drop_columns": config['dataset']['drop_columns']
        }
        logger.info("Retrieved feature information")
        return features
    except Exception as e:
        logger.error(f"Error getting features: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting features: {str(e)}"
        )
