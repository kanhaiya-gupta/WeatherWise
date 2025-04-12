import os
import shutil
from pathlib import Path

DATASET_TYPES = ["test", "train"]
DROP_COLNAMES = ["Date"]
TARGET_COLUMN = "RainTomorrow"
RAW_DATASET = "data/raw/weather.csv"
PROCESSED_DATASET = "data/processed/weather.csv"
REPORTS = "reports"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Define the path for the 'reports' folder
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
# Create the 'reports' directory if it doesn't exist
os.makedirs(REPORTS_DIR, exist_ok=True)


def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)