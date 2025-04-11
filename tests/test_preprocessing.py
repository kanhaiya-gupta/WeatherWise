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
