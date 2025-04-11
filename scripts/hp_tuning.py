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
