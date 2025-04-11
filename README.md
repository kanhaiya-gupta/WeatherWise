# WeatherWise: Automated Machine Learning Pipeline for Weather Prediction

**WeatherWise** is a machine learning project for predicting rainfall using a weather dataset. It leverages MLOps practices with **GitHub Actions** for CI/CD, **DVC** for data versioning, **FastAPI** for model deployment, and hyperparameter tuning for optimization.

## Directory Structure

- **data/**: Raw and processed datasets.
  - **raw/**: Unprocessed data (e.g., ).
  - **processed/**: Preprocessed data for training/testing.
- **notebooks/**: Jupyter notebooks for experimentation.
- **src/**: Source code.
  - **models/**: Model architecture ().
  - **preprocessing/**: Data cleaning ().
  - **training/**: Model training ().
  - **evaluation/**: Evaluation ().
  - **api/**: FastAPI application ().
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
- **tests/**: Unit tests (, ).
- **.github/workflows/**: CI/CD pipeline ().

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/WeatherWise.git
   cd WeatherWise
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize DVC**:
   ```bash
   dvc init
   dvc add data/raw/weather.csv
   ```

4. **Run the DVC pipeline**:
   ```bash
   dvc repro
   ```

## Running the FastAPI Server

1. Train the model (if not already done):
   ```bash
   dvc repro
   ```

2. Start the FastAPI server:
   ```bash
   python main.py
   ```

3. Access the API at . Endpoints:
   - : Predict rainfall probability.
   - : Check API health.

Example  command for prediction:
```bash
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
```

## Using Docker

1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

2. The API will be available at .

## CI/CD

The project uses GitHub Actions for CI/CD (see ). On push or pull request to :
- **Linting**: Runs  on source code.
- **Testing**: Runs  on unit tests.
- **Training**: Executes the DVC pipeline to train and tune the model.
- **Deployment**: Builds and pushes a Docker image to DockerHub (requires  and  secrets).

## Testing

Run unit tests with:
```bash
pytest tests/
```

## Requirements

- Python 3.8+
- DVC
- Docker (optional, for containerized deployment)
- GitHub Actions for CI/CD

## License

MIT License
