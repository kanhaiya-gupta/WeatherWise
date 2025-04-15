# WeatherWise: Automated Machine Learning Pipeline for Weather Prediction

**WeatherWise** is a comprehensive machine learning project designed to predict rainfall probability using a weather dataset. It integrates MLOps best practices, including **GitHub Actions** for CI/CD, **DVC** for data and model versioning, **FastAPI** for model deployment, and hyperparameter tuning for optimization. The project is structured to ensure reproducibility, scalability, and ease of deployment.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data & ML Pipeline Versioning with DVC](#data--ml-pipeline-versioning-with-dvc)
- [Running the FastAPI Server](#running-the-fastapi-server)
- [Using Docker](#using-docker)
- [CI/CD](#cicd)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features
- **Rainfall Prediction**: Uses a trained Random Forest model to predict rainfall probability based on weather features.
- **MLOps Pipeline**:
  - Data versioning and pipeline management with DVC.
  - Automated CI/CD workflows using GitHub Actions.
  - Hyperparameter tuning for model optimization.
- **FastAPI Deployment**:
  - RESTful API with interactive Swagger UI at `http://localhost:8000/docs`.
  - Endpoints for health checks and predictions.
- **Containerization**: Docker support for consistent development and deployment.
- **Testing**: Unit tests for preprocessing and model components.
- **Reporting**: Generates metrics, confusion matrices, and ROC curves.
- **Advanced Preprocessing**: Custom `WeatherDataPreprocessor` class for robust data handling.
- **Logging**: Comprehensive logging system for monitoring and debugging.
- **MLflow Integration**: Experiment tracking and model versioning.

## Project Structure

```
WeatherWise/
├── .dvc/                       # DVC configuration and cache
├── .github/workflows/          # GitHub Actions CI/CD workflows
│   ├── scripts.yml            # Scripts and linting workflow
│   ├── dvc.yml               # DVC pipeline workflow
│   └── docker.yml            # Docker build and push workflow
├── config/                     # Configuration files
│   ├── config.yaml            # Main configuration
│   └── hp_config.json         # Hyperparameter tuning settings
├── data/                       # Datasets
│   ├── raw/                    # Unprocessed data
│   │   └── weather.csv
│   └── processed/              # Preprocessed data
│       └── weather.csv
├── docs/                       # Documentation
│   ├── DVC_DRIVE.txt
│   ├── DVC_workflow_README.md
│   └── FastAPI_README.md
├── logs/                       # Application logs
├── models/                     # Trained models
│   └── weather_model.joblib
├── notebooks/                  # Jupyter notebooks for experimentation
├── reports/                    # Metrics and visualizations
│   ├── confusion_matrix.png
│   ├── hp_tuning_results.md
│   ├── metrics.json
│   ├── predictions.csv
│   ├── rfc_best_params.json
│   └── roc_curve.csv
├── scripts/                    # Automation scripts
│   ├── hp_tuning.py            # Hyperparameter tuning
│   └── metrics_and_plots.py    # Metrics and plot generation
├── src/                        # Source code
│   ├── api/                    # FastAPI application
│   │   └── app.py
│   ├── data/                   # Data processing
│   │   └── processor.py
│   ├── evaluation/             # Model evaluation
│   │   └── evaluate.py
│   ├── models/                 # Model definitions
│   │   └── model.py
│   ├── preprocessing/          # Data preprocessing
│   │   └── preprocess.py
│   ├── training/               # Model training
│   │   └── train.py
│   └── utils/                  # Utility functions
│       └── utils_and_constants.py
├── tests/                      # Unit tests
│   ├── test_model.py
│   └── test_preprocessing.py
├── .dvcignore
├── .env                        # Environment variables
├── .gitignore
├── activate_ml_env.sh          # Script to activate virtual environment
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Docker configuration
├── dvc.lock                    # DVC pipeline lock file
├── dvc.yaml                    # DVC pipeline definitions
├── main.py                     # FastAPI server entry point
├── requirements.txt            # Python dependencies
├── setup_weatherwise.sh        # Setup script
└── README.md                   # This file
```

## Prerequisites
- **Python**: 3.8 or higher
- **Git**: For cloning and version control
- **DVC**: For data and model versioning
- **Docker**: Optional, for containerized deployment
- **GitHub Account**: For CI/CD and optional DagsHub integration
- **Dependencies**: Listed in `requirements.txt` (e.g., `fastapi`, `uvicorn`, `pandas`, `joblib`, `pytest`, `dvc`, `mlflow`)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kanhaiya-gupta/WeatherWise.git
   cd WeatherWise
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   Alternatively, use the provided script:
   ```bash
   bash activate_ml_env.sh
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   - Create a `.env` file or modify the existing one to include paths like `MODEL_PATH` if needed.
   - Example:
     ```
     MODEL_PATH=models/weather_model.joblib
     MLFLOW_TRACKING_URI=mlruns
     ```

5. **Run Setup Script** (optional):
   ```bash
   bash setup_weatherwise.sh
   ```
   This script can automate environment setup if provided in the repository.

## Data & ML Pipeline Versioning with DVC
DVC is used to version datasets, models, and manage reproducible ML pipelines.

### Step-by-Step DVC Setup
1. **Initialize DVC**:
   ```bash
   dvc init
   git add .dvc .dvcignore
   git commit -m "Initialize DVC"
   ```

2. **Track Data**:
   Add the raw dataset:
   ```bash
   dvc add data/raw/weather.csv
   git add data/raw/weather.csv.dvc .gitignore
   git commit -m "Track raw dataset with DVC"
   ```

3. **Set Up Remote Storage** (e.g., DagsHub):
   ```bash
   dvc remote add -d origin https://dagshub.com/kanhaiya-gupta/WeatherWise.dvc
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user kanhaiya-gupta
   dvc remote modify origin --local password <your_dagshub_token>
   ```
   Push data to remote:
   ```bash
   dvc push
   ```

4. **Run the DVC Pipeline**:
   The `dvc.yaml` defines stages for preprocessing, training, and evaluation. Run:
   ```bash
   dvc repro
   ```
   This executes the pipeline, generating processed data, trained models, and reports.

## Running the FastAPI Server

1. **Start the Server**:
   ```bash
   python main.py
   ```
   The server will start at `http://localhost:8000`.

2. **Access the API**:
   - Interactive documentation: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`
   - Prediction endpoint: `http://localhost:8000/predict`

3. **Example Prediction Request**:
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{
          "Location": "Albury",
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
        }'
   ```

## Using Docker

1. **Build and Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

2. **Access the API**:
   The API will be available at `http://localhost:8000`.

## CI/CD

The project uses three separate GitHub Actions workflows:

1. **Scripts Workflow** (`scripts.yml`):
   - Runs linting and testing
   - Executes on pull requests and pushes to main

2. **DVC Pipeline Workflow** (`dvc.yml`):
   - Executes the DVC pipeline
   - Runs on schedule and manual triggers
   - Generates and stores model artifacts

3. **Docker Workflow** (`docker.yml`):
   - Builds and pushes Docker images
   - Runs on successful DVC pipeline completion
   - Requires DockerHub credentials

## Testing

1. **Run Unit Tests**:
   ```bash
   pytest tests/
   ```

2. **Test Coverage**:
   ```bash
   pytest --cov=src tests/
   ```

## Troubleshooting

1. **Model Loading Issues**:
   - Check the `MODEL_PATH` environment variable
   - Verify the model file exists at the specified path
   - Ensure the model file is not corrupted

2. **Preprocessing Errors**:
   - Verify all required features are present in the input data
   - Check the data types of input features
   - Ensure categorical features match the training data

3. **API Connection Issues**:
   - Verify the server is running
   - Check the port is not in use
   - Ensure all required environment variables are set

4. **DVC Pipeline Failures**:
   - Check the DVC cache
   - Verify remote storage configuration
   - Ensure all dependencies are installed

## License

MIT License
