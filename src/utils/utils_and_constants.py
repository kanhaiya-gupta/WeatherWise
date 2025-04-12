import os
import shutil
from pathlib import Path

DATASET_TYPES = ["test", "train"]
DROP_COLNAMES = ["Date"]
TARGET_COLUMN = "RainTomorrow"
RAW_DATASET = "data/raw/weather.csv"
PROCESSED_DATASET = "data/processed/weather.csv"
REPORTS = "reports"
MODELS = "models"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Define the path for the 'reports' folder
REPORTS_DIR = os.path.join(BASE_DIR, REPORTS)
MODEL_DIR = os.path.join(BASE_DIR, MODELS)
# Create the 'reports', 'Models' directory if it doesn't exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except:
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