# WeatherWise API

WeatherWise API is a FastAPI-based application that predicts the probability of rainfall based on weather features. It provides a simple and efficient interface to interact with a pre-trained machine learning model for rainfall prediction.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Testing the API](#testing-the-api)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features
- Predict rainfall probability using a trained Random Forest model.
- Interactive API documentation via Swagger UI at `http://localhost:8000/docs`.
- Health check endpoint to verify API status.
- Input validation using Pydantic models.
- Environment-agnostic model loading with error handling.

## Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment (recommended)
- Required Python packages (listed in `requirements.txt`)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kanhaiya-gupta/WeatherWise.git
   cd WeatherWise
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not present, install the required packages manually:
   ```bash
   pip install fastapi uvicorn pandas joblib
   ```

4. **Ensure Model Availability**:
   - Verify that the trained model file `models/trained_model_random_forest.pkl` exists in the `models/` directory.
   - If missing, retrain the model using the training scripts or download it if hosted elsewhere.

## Running the API
1. **Start the FastAPI Server**:
   From the project root directory (`WeatherWise/`), run:
   ```bash
   python main.py
   ```
   This starts the server on `http://localhost:8000` with auto-reload enabled for development.

2. **Access the API**:
   - Open `http://localhost:8000/docs` in a browser to access the interactive Swagger UI for testing endpoints.
   - Alternatively, use tools like `curl` or Postman to interact with the API.

## API Endpoints
- **GET `/health`**:
  - **Description**: Check the health status of the API and model.
  - **Response**:
    ```json
    {
      "status": "healthy",
      "model_loaded": true
    }
    ```
  - **Example**:
    ```bash
    curl http://localhost:8000/health
    ```

- **POST `/predict`**:
  - **Description**: Predict the probability of rainfall based on weather features.
  - **Request Body** (JSON):
    ```json
    {
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
    ```
  - **Response**:
    ```json
    {
      "rain_probability": 0.1234
    }
    ```
  - **Example**:
    ```bash
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"MinTemp": 13.4, "MaxTemp": 22.9, "Rainfall": 0.6, "WindGustSpeed": 44.0, "WindSpeed9am": 20.0, "WindSpeed3pm": 24.0, "Humidity9am": 71.0, "Humidity3pm": 22.0, "Pressure9am": 1007.7, "Pressure3pm": 1007.1, "Temp9am": 16.9, "Temp3pm": 21.8, "RainToday": 0}'
    ```

## Testing the API
- **Interactive Testing**:
  - Visit `http://localhost:8000/docs` to access the Swagger UI.
  - Use the UI to test the `/health` and `/predict` endpoints with sample inputs.
- **Command Line Testing**:
  - Use `curl` as shown in the [API Endpoints](#api-endpoints) section.
- **Postman**:
  - Import the endpoint URLs and JSON payloads into Postman for testing.

## Project Structure
```
WeatherWise/
├── main.py                   # Entry point to run the FastAPI server
├── src/
│   └── api/
│       └── app.py            # FastAPI application logic
├── models/
│   └── trained_model_random_forest.pkl  # Pre-trained model
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── ...                       # Other files (e.g., training scripts)
```

## Deployment
For production deployment, consider the following:
- **Disable Auto-Reload**:
  - Edit `main.py` to set `reload=False` and adjust `workers` (e.g., `workers=4`).
- **Use Gunicorn**:
  - Install Gunicorn:
    ```bash
    pip install gunicorn
    ```
  - Run the app:
    ```bash
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.app:app
    ```
- **Docker**:
  - Create a `Dockerfile`:
    ```dockerfile
    FROM python:3.9-slim

    WORKDIR /app

    COPY . .

    RUN pip install --no-cache-dir fastapi uvicorn pandas joblib

    EXPOSE 8000

    CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
    ```
  - Build and run:
    ```bash
    docker build -t weatherwise-api .
    docker run -p 8000:8000 weatherwise-api
    ```
- **Cloud Deployment**:
  - Deploy to platforms like Heroku, AWS, or GCP.
  - Set the `MODEL_PATH` environment variable if the model location differs.

## Troubleshooting
- **Model Not Found**:
  - Ensure `models/trained_model_random_forest.pkl` exists.
  - Check the `MODEL_PATH` environment variable or path in `src/api/app.py`.
- **Port Conflicts**:
  - If port `8000` is in use, change the port in `main.py` (e.g., `port=8001`).
- **Dependency Errors**:
  - Verify all packages are installed (`fastapi`, `uvicorn`, `pandas`, `joblib`).
  - Run `pip list` to check versions.
- **Prediction Failures**:
  - Ensure input JSON matches the `WeatherInput` schema (correct feature names and types).
  - Check server logs for detailed error messages.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.