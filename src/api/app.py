from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="WeatherWise API")

# Define input schema
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

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "../../models/best_model.pkl")
model = joblib.load(MODEL_PATH)

@app.post("/predict")
async def predict_rainfall(data: WeatherInput):
    """Predict rainfall probability."""
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict()])
        prediction = model.predict_proba(input_data)[:, 1][0]  # Probability of rain
        return {"rain_probability": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check API health."""
    return {"status": "healthy"}
