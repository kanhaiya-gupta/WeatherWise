import os
import shutil
import pandas as pd
from pathlib import Path
import yaml

# Define paths relative to the base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')
CONFIG_PATH_HP = os.path.join(BASE_DIR, 'config', 'hp_config.json')

# Load configuration from YAML file
def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

# Read configuration values
config = load_config()

# Assign configuration values to constants
DATASET_TYPES = config.get('dataset_types', ['test', 'train'])
DROP_COLNAMES = config.get('drop_colnames', ['Date'])
TARGET_COLUMN = config['dataset']['target_column']  # Get target column directly from dataset config
RAW_DATASET = os.path.join(BASE_DIR, config.get('preprocessing', {}).get('raw_data_path', 'data/raw/weather.csv'))
PROCESSED_DATASET = os.path.join(BASE_DIR, config.get('preprocessing', {}).get('processed_data_path', 'data/processed/weather.csv'))
REPORTS = config.get('evaluation', {}).get('reports_dir', 'reports')
MODELS = config.get('training', {}).get('model_save_path', 'models').rsplit('/', 1)[0]  # Extract directory
MODEL_PATH = os.path.join(BASE_DIR, config.get('training', {}).get('model_save_path', 'models/trained_model_random_forest.pkl'))
MLFLOW_TRACKING_URI = config.get('mlflow', {}).get('tracking_uri', 'http://localhost:5000')
MLFLOW_EXPERIMENT_NAME = config.get('training', {}).get('experiment_name', 'weather_prediction')
BEST_PAR = config.get('best_par', 'rfc_best_params.json')

# Define directories
REPORTS_DIR = os.path.join(BASE_DIR, REPORTS)
MODEL_DIR = os.path.join(BASE_DIR, MODELS)

# Create the 'reports' and 'models' directories if they don't exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_PAR_PATH = os.path.join(REPORTS_DIR, BEST_PAR)

def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)

def get_hp_tuning_results(grid_search):
    """
    Generate a markdown table from GridSearchCV results.

    Parameters:
    - grid_search: Fitted GridSearchCV object

    Returns:
    - str: Markdown table string
    """
    results = pd.DataFrame(grid_search.cv_results_)
    # Select relevant columns
    columns = ['param_' + key for key in grid_search.param_grid.keys()] + ['mean_test_score', 'std_test_score', 'rank_test_score']
    results = results[columns].sort_values('rank_test_score')

    # Create markdown table
    header = "| " + " | ".join([col.replace("param_", "") for col in columns]) + " |"
    separator = "| " + " | ".join(["---" for _ in columns]) + " |"
    rows = []
    for _, row in results.iterrows():
        row_values = [f"{row[col]:.4f}" if col in ['mean_test_score', 'std_test_score'] else str(row[col]) for col in columns]
        rows.append("| " + " | ".join(row_values) + " |")

    markdown_table = "\n".join([header, separator] + rows)
    return markdown_table