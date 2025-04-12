# import pandas as pd
# import json
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV

# def hyperparameter_tuning(train_X_path, train_y_path, config_path, output_model_path):
#     """Perform hyperparameter tuning and save the best model."""
#     X_train = pd.read_csv(train_X_path)
#     y_train = pd.read_csv(train_y_path).values.ravel()
    
#     with open(config_path, 'r') as f:
#         param_grid = json.load(f)['model']
    
#     model = RandomForestClassifier(random_state=42)
#     grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
#     grid_search.fit(X_train, y_train)
    
#     print("Best parameters:", grid_search.best_params_)
#     print("Best score:", grid_search.best_score_)
    
#     joblib.dump(grid_search.best_estimator_, output_model_path)
#     print(f"Best model saved to {output_model_path}")

# if __name__ == "__main__":
#     hyperparameter_tuning('../data/processed/train_X.csv', '../data/processed/train_y.csv', 
#                          '../config/hp_config.json', '../models/best_model.pkl')
    

import os
import json
import joblib # type: ignore
import pandas as pd # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import GridSearchCV, train_test_split # type: ignore
from src.utils.utils_and_constants import BASE_DIR, REPORTS_DIR, CONFIG_PATH # type: ignore
from src.utils.utils_and_constants import PROCESSED_DATASET, TARGET_COLUMN, get_hp_tuning_results # type: ignore

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y

def main():
    X, y = load_data(PROCESSED_DATASET)
    X_train, _, y_train, _ = train_test_split(X, y, random_state=1993)

    model = RandomForestClassifier()
    # Read the config file to define the hyperparameter search space
    param_grid = json.load(open(CONFIG_PATH, "r"))

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
