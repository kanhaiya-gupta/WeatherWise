#!/bin/bash

# Script to create the WeatherWise project with Python files, CI/CD, FastAPI, and more

# Define the project root directory
PROJECT_DIR="WeatherWise"

# Create the main project directory
echo "Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"

# Create subdirectories
echo "Creating subdirectories..."
mkdir -p "$PROJECT_DIR/data/raw"
mkdir -p "$PROJECT_DIR/data/processed"
mkdir -p "$PROJECT_DIR/notebooks"
mkdir -p "$PROJECT_DIR/src/models"
mkdir -p "$PROJECT_DIR/src/preprocessing"
mkdir -p "$PROJECT_DIR/src/training"
mkdir -p "$PROJECT_DIR/src/evaluation"
mkdir -p "$PROJECT_DIR/src/api"
mkdir -p "$PROJECT_DIR/config"
mkdir -p "$PROJECT_DIR/reports"
mkdir -p "$PROJECT_DIR/scripts"
mkdir -p "$PROJECT_DIR/tests"
mkdir -p "$PROJECT_DIR/.github/workflows"

# Create weather.csv in data/raw
echo "Creating weather.csv..."
cat <<EOL > "$PROJECT_DIR/data/raw/weather.csv"
Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday,RISK_MM,RainTomorrow
2008-12-01,Albury,13.4,22.9,0.6,,,W,44.0,W,WNW,20.0,24.0,71.0,22.0,1007.7,1007.1,8.0,,16.9,21.8,No,0.0,No
2008-12-02,Albury,7.4,25.1,0.0,,,WNW,44.0,NNW,WSW,4.0,22.0,44.0,25.0,1010.6,1007.8,,,17.2,24.3,No,0.0,No
2008-12-03,Albury,12.9,25.7,0.0,,,WSW,46.0,W,WSW,19.0,26.0,38.0,30.0,1007.6,1008.7,,2.0,21.0,23.2,No,0.0,No
EOL

# Create Python files
echo "Creating Python files..."

# src/preprocessing/preprocess.py
cat <<EOL > "$PROJECT_DIR/src/preprocessing/preprocess.py"
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data(input_path, output_dir):
    """Preprocess weather data and save train/test splits."""
    df = pd.read_csv(input_path)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Convert categorical to numeric
    df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
    df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})
    
    # Drop columns with excessive missing data or irrelevant
    df = df.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'WindGustDir', 
                  'WindDir9am', 'WindDir3pm', 'Cloud9am', 'Cloud3pm'], axis=1)
    
    # Features and target
    X = df.drop('RainTomorrow', axis=1)
    y = df['RainTomorrow']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, 'train_X.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'test_X.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'train_y.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'test_y.csv'), index=False)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data('../data/raw/weather.csv', '../data/processed')
EOL

# src/models/model.py
cat <<EOL > "$PROJECT_DIR/src/models/model.py"
from sklearn.ensemble import RandomForestClassifier

def define_model():
    """Define the machine learning model."""
    model = RandomForestClassifier(random_state=42)
    return model
EOL

# src/training/train.py
cat <<EOL > "$PROJECT_DIR/src/training/train.py"
import pandas as pd
import joblib
from src.models.model import define_model
import os

def train_model(train_X_path, train_y_path, model_path):
    """Train the model and save it."""
    X_train = pd.read_csv(train_X_path)
    y_train = pd.read_csv(train_y_path).values.ravel()
    
    model = define_model()
    model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model('../data/processed/train_X.csv', '../data/processed/train_y.csv', '../models/model.pkl')
EOL

# src/evaluation/evaluate.py
cat <<EOL > "$PROJECT_DIR/src/evaluation/evaluate.py"
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

def evaluate_model(test_X_path, test_y_path, model_path, output_dir):
    """Evaluate the model and save metrics/plots."""
    X_test = pd.read_csv(test_X_path)
    y_test = pd.read_csv(test_y_path).values.ravel()
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
    }
    
    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Metrics saved to {output_dir}/metrics.json")
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model('../data/processed/test_X.csv', '../data/processed/test_y.csv', 
                   '../models/model.pkl', '../reports')
EOL

# scripts/hp_tuning.py
cat <<EOL > "$PROJECT_DIR/scripts/hp_tuning.py"
import pandas as pd
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def hyperparameter_tuning(train_X_path, train_y_path, config_path, output_model_path):
    """Perform hyperparameter tuning and save the best model."""
    X_train = pd.read_csv(train_X_path)
    y_train = pd.read_csv(train_y_path).values.ravel()
    
    with open(config_path, 'r') as f:
        param_grid = json.load(f)['model']
    
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    
    joblib.dump(grid_search.best_estimator_, output_model_path)
    print(f"Best model saved to {output_model_path}")

if __name__ == "__main__":
    hyperparameter_tuning('../data/processed/train_X.csv', '../data/processed/train_y.csv', 
                         '../config/hp_config.json', '../models/best_model.pkl')
EOL

# scripts/metrics_and_plots.py
cat <<EOL > "$PROJECT_DIR/scripts/metrics_and_plots.py"
import json
import matplotlib.pyplot as plt

def generate_metrics_plot(metrics_path, output_path):
    """Generate a bar plot of metrics."""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.savefig(output_path)
    plt.close()
    print(f"Metrics plot saved to {output_path}")

if __name__ == "__main__":
    generate_metrics_plot('../reports/metrics.json', '../reports/metrics_bar.png')
EOL

# src/api/app.py
cat <<EOL > "$PROJECT_DIR/src/api/app.py"
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
EOL

# main.py (FastAPI entry point)
cat <<EOL > "$PROJECT_DIR/main.py"
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
EOL

# tests/test_preprocessing.py
cat <<EOL > "$PROJECT_DIR/tests/test_preprocessing.py"
import pandas as pd
import os
from src.preprocessing.preprocess import preprocess_data

def test_preprocessing():
    """Test data preprocessing."""
    # Create temporary input file
    input_path = "data/raw/test_weather.csv"
    output_dir = "data/processed"
    df = pd.DataFrame({
        'Date': ['2008-12-01'],
        'Location': ['Albury'],
        'MinTemp': [13.4],
        'MaxTemp': [22.9],
        'Rainfall': [0.6],
        'Evaporation': [None],
        'Sunshine': [None],
        'WindGustDir': ['W'],
        'WindGustSpeed': [44.0],
        'WindDir9am': ['W'],
        'WindDir3pm': ['WNW'],
        'WindSpeed9am': [20.0],
        'WindSpeed3pm': [24.0],
        'Humidity9am': [71.0],
        'Humidity3pm': [22.0],
        'Pressure9am': [1007.7],
        'Pressure3pm': [1007.1],
        'Cloud9am': [8.0],
        'Cloud3pm': [None],
        'Temp9am': [16.9],
        'Temp3pm': [21.8],
        'RainToday': ['No'],
        'RISK_MM': [0.0],
        'RainTomorrow': ['No']
    })
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    df.to_csv(input_path, index=False)
    
    # Run preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(input_path, output_dir)
    
    # Assertions
    assert not X_train.empty, "Training features should not be empty"
    assert not y_test.empty, "Test labels should not be empty"
    assert os.path.exists(os.path.join(output_dir, 'train_X.csv')), "Train features file missing"
    
    # Clean up
    os.remove(input_path)
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))
EOL

# tests/test_model.py
cat <<EOL > "$PROJECT_DIR/tests/test_model.py"
from src.models.model import define_model

def test_model_definition():
    """Test model definition."""
    model = define_model()
    assert model is not None, "Model should be defined"
    assert hasattr(model, 'fit'), "Model should have a fit method"
EOL

# Configuration files
echo "Creating configuration files..."

# config/hp_config.json
cat <<EOL > "$PROJECT_DIR/config/hp_config.json"
{
  "model": {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
  }
}
EOL

# config/dvc.yaml
cat <<EOL > "$PROJECT_DIR/config/dvc.yaml"
stages:
  preprocess:
    cmd: python src/preprocessing/preprocess.py
    deps:
      - data/raw/weather.csv
    outs:
      - data/processed/train_X.csv
      - data/processed/test_X.csv
      - data/processed/train_y.csv
      - data/processed/test_y.csv
  train:
    cmd: python src/training/train.py
    deps:
      - data/processed/train_X.csv
      - data/processed/train_y.csv
      - src/models/model.py
    outs:
      - models/model.pkl
  tune:
    cmd: python scripts/hp_tuning.py
    deps:
      - data/processed/train_X.csv
      - data/processed/train_y.csv
      - config/hp_config.json
    outs:
      - models/best_model.pkl
  evaluate:
    cmd: python src/evaluation/evaluate.py
    deps:
      - data/processed/test_X.csv
      - data/processed/test_y.csv
      - models/best_model.pkl
    outs:
      - reports/metrics.json
      - reports/confusion_matrix.png
EOL

# CI/CD Workflow
echo "Creating GitHub Actions workflow..."

# .github/workflows/ci.yml
cat <<EOL > "$PROJECT_DIR/.github/workflows/ci.yml"
name: WeatherWise CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install flake8
      - name: Run linting
        run: flake8 src scripts tests

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt pytest
      - name: Run tests
        run: pytest tests/

  train:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Install DVC
        run: pip install dvc
      - name: Run DVC pipeline
        run: |
          dvc init --no-scm
          dvc repro
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: model
          path: models/best_model.pkl

  deploy:
    runs-on: ubuntu-latest
    needs: [train]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Download model artifact
        uses: actions/download-artifact@v3
        with:
          name: model
          path: models/
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      - name: Login to DockerHub
        uses: docker/login-action@v2
        with:
          username: \${{ secrets.DOCKER_USERNAME }}
          password: \${{ secrets.DOCKER_PASSWORD }}
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: \${{ secrets.DOCKER_USERNAME }}/weatherwise:latest
EOL

# Additional files
echo "Creating additional files..."

# requirements.txt
cat <<EOL > "$PROJECT_DIR/requirements.txt"
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.0.3
scikit-learn==1.3.2
joblib==1.3.2
dvc==3.30.1
matplotlib==3.7.3
seaborn==0.13.0
pytest==7.4.3
flake8==6.1.0
python-dotenv==1.0.0
EOL

# Dockerfile
cat <<EOL > "$PROJECT_DIR/Dockerfile"
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
EOL

# docker-compose.yml
cat <<EOL > "$PROJECT_DIR/docker-compose.yml"
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      - MODEL_PATH=/app/models/best_model.pkl
    depends_on:
      - train

  train:
    build: .
    command: bash -c "pip install dvc && dvc repro"
    volumes:
      - .:/app
EOL

# .env
cat <<EOL > "$PROJECT_DIR/.env"
MODEL_PATH=models/best_model.pkl
EOL

# .gitignore
cat <<EOL > "$PROJECT_DIR/.gitignore"
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt

# Data
data/processed/*
!data/processed/.gitkeep
data/raw/*
!data/raw/weather.csv
!data/raw/.gitkeep

# Reports
reports/*
!reports/metrics.json
!reports/confusion_matrix.png
!reports/metrics_bar.png

# Notebooks
notebooks/*.ipynb
!notebooks/.gitkeep

# Models
models/*
!models/.gitkeep

# Tests
tests/__pycache__/

# Docker
*.log
.DS_Store
.env

# DVC
.dvc/
.dvcignore
EOL

# .dvcignore
cat <<EOL > "$PROJECT_DIR/.dvcignore"
data/processed/*
reports/*
models/*
*.log
*.tmp
EOL

# README.md
cat <<EOL > "$PROJECT_DIR/README.md"
# WeatherWise: Automated Machine Learning Pipeline for Weather Prediction

**WeatherWise** is a machine learning project for predicting rainfall using a weather dataset. It leverages MLOps practices with **GitHub Actions** for CI/CD, **DVC** for data versioning, **FastAPI** for model deployment, and hyperparameter tuning for optimization.

## Directory Structure

- **data/**: Raw and processed datasets.
  - **raw/**: Unprocessed data (e.g., `weather.csv`).
  - **processed/**: Preprocessed data for training/testing.
- **notebooks/**: Jupyter notebooks for experimentation.
- **src/**: Source code.
  - **models/**: Model architecture (`model.py`).
  - **preprocessing/**: Data cleaning (`preprocess.py`).
  - **training/**: Model training (`train.py`).
  - **evaluation/**: Evaluation (`evaluate.py`).
  - **api/**: FastAPI application (`app.py`).
- **config/**: Configuration files.
  - **hp_config.json**: Hyperparameter tuning settings.
  - **dvc.yaml**: DVC pipeline definitions.
- **reports/**: Metrics and visualizations.
  - **metrics.json**: Model performance metrics.
  - **confusion_matrix.png**: Confusion matrix plot.
  - **metrics_bar.png**: Metrics bar plot.
- **scripts/**: Automation scripts.
  - **hp_tuning.py**: Hyperparameter tuning.
  - **metrics_and_plots.py**: Metrics and plot generation.
- **tests/**: Unit tests (`test_preprocessing.py`, `test_model.py`).
- **.github/workflows/**: CI/CD pipeline (`ci.yml`).

## Setup

1. **Clone the repository**:
   \`\`\`bash
   git clone https://github.com/<your-username>/WeatherWise.git
   cd WeatherWise
   \`\`\`

2. **Install dependencies**:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Initialize DVC**:
   \`\`\`bash
   dvc init
   dvc add data/raw/weather.csv
   \`\`\`

4. **Run the DVC pipeline**:
   \`\`\`bash
   dvc repro
   \`\`\`

## Running the FastAPI Server

1. Train the model (if not already done):
   \`\`\`bash
   dvc repro
   \`\`\`

2. Start the FastAPI server:
   \`\`\`bash
   python main.py
   \`\`\`

3. Access the API at `http://localhost:8000`. Endpoints:
   - `POST /predict`: Predict rainfall probability.
   - `GET /health`: Check API health.

Example `curl` command for prediction:
\`\`\`bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
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
}'
\`\`\`

## Using Docker

1. Build and run with Docker Compose:
   \`\`\`bash
   docker-compose up --build
   \`\`\`

2. The API will be available at `http://localhost:8000`.

## CI/CD

The project uses GitHub Actions for CI/CD (see `.github/workflows/ci.yml`). On push or pull request to `main`:
- **Linting**: Runs `flake8` on source code.
- **Testing**: Runs `pytest` on unit tests.
- **Training**: Executes the DVC pipeline to train and tune the model.
- **Deployment**: Builds and pushes a Docker image to DockerHub (requires `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets).

## Testing

Run unit tests with:
\`\`\`bash
pytest tests/
\`\`\`

## Requirements

- Python 3.8+
- DVC
- Docker (optional, for containerized deployment)
- GitHub Actions for CI/CD

## License

MIT License
EOL

# Create .gitkeep files for empty directories
touch "$PROJECT_DIR/data/processed/.gitkeep"
touch "$PROJECT_DIR/notebooks/.gitkeep"
touch "$PROJECT_DIR/models/.gitkeep"

# Initialize Git repository
echo "Initializing Git repository..."
cd "$PROJECT_DIR"
git init
git add .
git commit -m "Initial commit: Set up WeatherWise with Python files, CI/CD, FastAPI, and Docker"

# Final message
echo "Project structure for $PROJECT_DIR created successfully!"
echo "Navigate to $PROJECT_DIR and start working."
echo "To run the FastAPI server: python main.py"
echo "To run with Docker: docker-compose up --build"
echo "To execute the DVC pipeline: dvc init && dvc repro"