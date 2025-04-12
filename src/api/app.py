# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# import os

# app = FastAPI(title="WeatherWise API")

# # Define input schema
# class WeatherInput(BaseModel):
#     MinTemp: float
#     MaxTemp: float
#     Rainfall: float
#     WindGustSpeed: float
#     WindSpeed9am: float
#     WindSpeed3pm: float
#     Humidity9am: float
#     Humidity3pm: float
#     Pressure9am: float
#     Pressure3pm: float
#     Temp9am: float
#     Temp3pm: float
#     RainToday: int

# # Load model
# MODEL_PATH = os.getenv("MODEL_PATH", "../../models/trained_model_random_forest.pkl")
# model = joblib.load(MODEL_PATH)

# @app.post("/predict")
# async def predict_rainfall(data: WeatherInput):
#     """Predict rainfall probability."""
#     try:
#         # Convert input to DataFrame
#         input_data = pd.DataFrame([data.dict()])
#         prediction = model.predict_proba(input_data)[:, 1][0]  # Probability of rain
#         return {"rain_probability": float(prediction)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/health")
# async def health_check():
#     """Check API health."""
#     return {"status": "healthy"}

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
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    WindGustSpeed: float
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Temp9am: float
    Temp3pm: float
    RainToday: int

    class Config:
        schema_extra = {
            "example": {
                "MinTemp": 13.4,
                "MaxTemp": 22.9,
                "Rainfall": 0.6,
                "WindGustSpeed": 44.0,
                "WindSpeed9am": 20.0,
                "WindSpeed3pm": 24.0,
                "Humidity9am": 71.0,
                "Humidity3pm": 22.0,
                "Pressure9am": 1007.7,
                "Pressure3pm": 1007.1,
                "Temp9am": 16.9,
                "Temp3pm": 21.8,
                "RainToday": 0
            }
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
