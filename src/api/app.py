from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(
    title="WeatherWise API",
    description="API for predicting rainfall probability based on weather features",
    version="1.0.0"
)

# Define input schema with constraints
class WeatherInput(BaseModel):
    Location: float
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    Evaporation: float
    Sunshine: float
    WindGustDir: float
    WindGustSpeed: float
    WindDir9am: float
    WindDir3pm: float
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Cloud9am: float
    Cloud3pm: float
    Temp9am: float
    Temp3pm: float
    RainToday: int  # 0 for No, 1 for Yes
    RISK_MM: float

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "Location": 0.543,
                    "MinTemp": 13.4,
                    "MaxTemp": 22.9,
                    "Rainfall": 0.6,
                    "Evaporation": 4.5,
                    "Sunshine": 6.7,
                    "WindGustDir": 0.61,
                    "WindGustSpeed": 44.0,
                    "WindDir9am": 0.48,
                    "WindDir3pm": 0.55,
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
                    "RainToday": 0,
                    "RISK_MM": 0.0
                }
            ]
        }
# Load model with error handling
try:
    MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "../../models/trained_model_random_forest.pkl"))
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

@app.post("/predict")
async def predict_rainfall(data: WeatherInput):
    """Predict rainfall probability based on input weather features."""
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict()])
        # Ensure feature order matches model's expectations
        prediction = model.predict_proba(input_data)[:, 1][0]  # Probability of rain
        return {"rain_probability": round(float(prediction), 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Check API health status."""
    return {"status": "healthy", "model_loaded": bool(model)}
