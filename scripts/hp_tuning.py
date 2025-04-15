from typing import Dict, Any, List
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
import mlflow
import yaml
from pathlib import Path
import json
import os
import joblib # type: ignore
import pandas as pd # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import GridSearchCV, train_test_split # type: ignore
from src.utils.utils_and_constants import REPORTS_DIR, CONFIG_PATH_HP # type: ignore
from src.utils.utils_and_constants import PROCESSED_DATASET, TARGET_COLUMN, get_hp_tuning_results # type: ignore

class HyperparameterTuner:
    """Class for hyperparameter tuning of weather models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tuner with configuration.
        
        Args:
            config: Dictionary containing tuning configuration
                - param_distributions: Parameter distributions for random search
                - n_iter: Number of parameter settings to try
                - cv: Number of cross-validation folds
                - random_state: Random seed for reproducibility
                - scoring: Metric to optimize
                - n_jobs: Number of parallel jobs
        """
        self.config = config
        
    def tune(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        run_name: str = "hyperparameter_tuning"
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        
        Args:
            model: Base model to tune
            X: Feature matrix
            y: Target vector
            run_name: Name for MLflow run
            
        Returns:
            Dictionary containing best parameters and scores
        """
        with mlflow.start_run(run_name=run_name):
            # Create random search object
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=self.config['param_distributions'],
                n_iter=self.config['n_iter'],
                cv=self.config['cv'],
                random_state=self.config['random_state'],
                scoring=self.config['scoring'],
                n_jobs=self.config.get('n_jobs', -1)
            )
            
            # Fit random search
            random_search.fit(X, y)
            
            # Log parameters and metrics
            mlflow.log_params({
                "n_iter": self.config['n_iter'],
                "cv_folds": self.config['cv'],
                "scoring": self.config['scoring']
            })
            
            mlflow.log_metrics({
                "best_score": random_search.best_score_,
                "mean_cv_score": random_search.cv_results_['mean_test_score'].mean(),
                "std_cv_score": random_search.cv_results_['std_test_score'].mean()
            })
            
            # Save best parameters
            best_params = random_search.best_params_
            results = {
                "best_params": best_params,
                "best_score": random_search.best_score_,
                "cv_results": {
                    "mean_test_score": random_search.cv_results_['mean_test_score'].tolist(),
                    "std_test_score": random_search.cv_results_['std_test_score'].tolist()
                }
            }
            
            # Save results to file
            output_path = Path(self.config['output_path'])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            # Log results file as artifact
            mlflow.log_artifact(str(output_path))
            
            return results
            
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y

def main():
    # Load configuration
    with open('config/hp_config.json', 'r') as f:
        config = json.load(f)
        
    # Initialize tuner
    tuner = HyperparameterTuner(config)
    
    # Load data and model (implement as needed)
    # X, y = load_data()
    # model = create_base_model()
    
    # Perform tuning
    # results = tuner.tune(model, X, y)
    
    X, y = load_data(PROCESSED_DATASET)
    X_train, _, y_train, _ = train_test_split(X, y, random_state=1993)

    model = RandomForestClassifier()
    # Read the config file to define the hyperparameter search space
    param_grid = json.load(open(CONFIG_PATH_HP, "r"))

    # Perform Grid Search Cross Validation on training data
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    print("====================Best Hyperparameters==================")
    print(json.dumps(best_params, indent=2))
    print("==========================================================")

    # Define the full path for the metrics file
    metrics_file = os.path.join(REPORTS_DIR, 'rfc_best_params.json')
    with open(metrics_file, "w") as outfile:
        json.dump(best_params, outfile)

    markdown_table = get_hp_tuning_results(grid_search)
    markdown_file_path = os.path.join(REPORTS_DIR, 'hp_tuning_results.md')
    with open(markdown_file_path, "w") as markdown_file:
        markdown_file.write(markdown_table)

if __name__ == "__main__":
    main()
