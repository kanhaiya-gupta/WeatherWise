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
