from typing import List, Tuple, Dict, Any
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import mlflow
from pathlib import Path
import numpy as np
import yaml
import logging

from src.utils.utils_and_constants import (
    DROP_COLNAMES,
    PROCESSED_DATASET,
    RAW_DATASET,
    TARGET_COLUMN,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeatherDataPreprocessor:
    """Class for preprocessing weather data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Dictionary containing preprocessing configuration
                - dataset: Dictionary containing dataset configuration
                    - raw_data_path: Path to raw data
                    - processed_data_path: Path to save processed data
                    - feature_columns: List of feature column names
                    - target_column: Name of target column
                    - categorical_columns: List of categorical column names
                    - drop_columns: List of columns to drop
                - mlflow: Dictionary containing MLflow configuration
                    - tracking_uri: MLflow tracking URI
                    - experiment_name: MLflow experiment name
        """
        self.config = config
        self.dataset_config = config['dataset']
        self.mlflow_config = config['mlflow']
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="mean")
        
        # Set configuration parameters
        self.feature_columns = self.dataset_config.get('feature_columns', [])
        self.target_column = self.dataset_config.get('target_column', TARGET_COLUMN)
        self.categorical_columns = self.dataset_config.get('categorical_columns', [])
        self.drop_columns = self.dataset_config.get('drop_columns', DROP_COLNAMES)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        mlflow.set_experiment(self.mlflow_config['experiment_name'])
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from specified path."""
        try:
            # Try to get path from dataset paths
            if 'paths' in self.dataset_config and 'raw' in self.dataset_config['paths']:
                data_path = Path(self.dataset_config['paths']['raw'])
            # Fallback to raw_data_path
            elif 'raw_data_path' in self.dataset_config:
                data_path = Path(self.dataset_config['raw_data_path'])
            else:
                raise KeyError("No data path found in config. Need either 'paths.raw' or 'raw_data_path'")
                
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found at {data_path}")
                
            df = pd.read_csv(data_path)
            logger.info(f"Successfully loaded data from {data_path}")
            
            if self.drop_columns:
                df = df.drop(columns=self.drop_columns, errors='ignore')
                logger.info(f"Dropped columns: {self.drop_columns}")
                
            # Convert target column to binary if needed
            if self.target_column in df.columns:
                df[self.target_column] = df[self.target_column].map({"Yes": 1, "No": 0})
                logger.info(f"Converted target column '{self.target_column}' to binary")
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        missing_before = df.isnull().sum().sum()
        df = df.ffill().bfill()
        missing_after = df.isnull().sum().sum()
        logger.info(f"Handled {missing_before - missing_after} missing values")
        return df
        
    def encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Target encode categorical features.
        
        Args:
            df: DataFrame to encode
            is_training: Whether this is training data (has target values)
        """
        if not self.categorical_columns:
            return df
            
        encoded_data = df.copy()
        
        # First convert target to numeric if it exists
        if self.target_column in df.columns:
            encoded_data[self.target_column] = encoded_data[self.target_column].map({"Yes": 1, "No": 0})
            
        for col in self.categorical_columns:
            if col in df.columns:
                if is_training and self.target_column in df.columns:
                    # For training data, calculate mean target value for each category
                    encoding_map = encoded_data.groupby(col)[self.target_column].mean().to_dict()
                    # Store the encoding map for later use
                    if not hasattr(self, 'encoding_maps'):
                        self.encoding_maps = {}
                    self.encoding_maps[col] = encoding_map
                else:
                    # For new data, use the stored encoding map
                    if not hasattr(self, 'encoding_maps') or col not in self.encoding_maps:
                        raise ValueError(f"No encoding map found for categorical feature: {col}")
                    encoding_map = self.encoding_maps[col]
                
                # Apply encoding
                encoded_data[col] = encoded_data[col].map(encoding_map)
                logger.info(f"Encoded categorical feature: {col}")
                
        return encoded_data
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from the data."""
        original_features = set(df.columns)
        
        # Example feature engineering
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['heat_index'] = 0.5 * (df['temperature'] + 61.0 + 
                                    ((df['temperature']-68.0)*1.2) + 
                                    (df['humidity']*0.094))
            
        if 'wind_speed' in df.columns and 'temperature' in df.columns:
            df['wind_chill'] = 35.74 + (0.6215 * df['temperature']) - \
                              (35.75 * (df['wind_speed']**0.16)) + \
                              (0.4275 * df['temperature'] * (df['wind_speed']**0.16))
                              
        new_features = set(df.columns) - original_features
        if new_features:
            logger.info(f"Created new features: {new_features}")
            
        return df
        
    def scale_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features using StandardScaler."""
        try:
            # First encode categorical features (training data)
            df = self.encode_categorical_features(df, is_training=True)
            
            # Select features and target
            X = df[self.feature_columns]
            y = df[self.target_column] if self.target_column in df.columns else None
            
            # Debug logging
            logger.info(f"Expected features: {self.feature_columns}")
            logger.info(f"Available features: {list(X.columns)}")
            logger.info(f"Numeric features: {[col for col in X.columns if col not in self.categorical_columns]}")
            logger.info(f"Categorical features: {[col for col in X.columns if col in self.categorical_columns]}")
            
            # Handle numeric features
            numeric_features = [col for col in X.columns if col not in self.categorical_columns]
            if numeric_features:
                X_numeric = X[numeric_features]
                # Convert to numeric type
                X_numeric = X_numeric.apply(pd.to_numeric, errors='coerce')
                # Impute and scale numeric features
                X_numeric_imputed = self.imputer.fit_transform(X_numeric)
                
                # Debug logging
                logger.info(f"Shape after imputation: {X_numeric_imputed.shape}")
                
                X_numeric_scaled = pd.DataFrame(
                    self.scaler.fit_transform(X_numeric_imputed),
                    columns=numeric_features,
                    index=X.index
                )
                
                # Debug logging
                logger.info(f"Shape after scaling: {X_numeric_scaled.shape}")
            else:
                X_numeric_scaled = pd.DataFrame(index=X.index)
            
            # Initialize result DataFrame with original index
            X_scaled = pd.DataFrame(index=X.index)
            
            # Add scaled numeric features
            if numeric_features:
                X_scaled = pd.concat([X_scaled, X_numeric_scaled], axis=1)
            
            # Handle categorical features (already encoded)
            categorical_features = [col for col in X.columns if col in self.categorical_columns]
            if categorical_features:
                X_categorical = X[categorical_features]
                # Convert to numeric type
                X_categorical = X_categorical.apply(pd.to_numeric, errors='coerce')
                X_scaled = pd.concat([X_scaled, X_categorical], axis=1)
            
            # Debug logging
            logger.info(f"Final shape before reordering: {X_scaled.shape}")
            logger.info(f"Final columns before reordering: {list(X_scaled.columns)}")
            
            # Reorder columns to match original feature order
            X_scaled = X_scaled[self.feature_columns]
            
            # Debug logging
            logger.info(f"Final shape after reordering: {X_scaled.shape}")
            logger.info(f"Final columns after reordering: {list(X_scaled.columns)}")
            
            return X_scaled, y
            
        except KeyError as e:
            raise ValueError(f"Missing required features: {str(e)}")
        except Exception as e:
            logger.error(f"Error during feature scaling: {e}")
            raise
        
    def process(self, data: pd.DataFrame = None, save_processed: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the full preprocessing pipeline.
        
        Args:
            data: Optional DataFrame to process. If None, will load from raw_data_path
            save_processed: Whether to save processed data
            
        Returns:
            Tuple of (processed features, target)
        """
        with mlflow.start_run(nested=True):
            try:
                # Log preprocessing parameters
                mlflow.log_params({
                    "feature_columns": self.feature_columns,
                    "target_column": self.target_column,
                    "categorical_columns": self.categorical_columns,
                    "drop_columns": self.drop_columns
                })
                
                # Load or use provided data
                if data is None:
                    df = self.load_data()
                else:
                    df = data.copy()
                    
                # Apply preprocessing steps
                df = self.handle_missing_values(df)
                df = self.create_features(df)
                X, y = self.scale_features(df)
                
                # Log metrics about the preprocessing
                mlflow.log_metrics({
                    "n_features": len(self.feature_columns),
                    "n_samples": len(df),
                    "missing_values_pct": (df.isnull().sum().sum() / df.size) * 100
                })
                
                if save_processed:
                    try:
                        # Try to get path from dataset paths
                        if 'paths' in self.dataset_config and 'processed' in self.dataset_config['paths']:
                            output_path = Path(self.dataset_config['paths']['processed'])
                        # Fallback to processed_data_path
                        elif 'processed_data_path' in self.dataset_config:
                            output_path = Path(self.dataset_config['processed_data_path'])
                        else:
                            logger.warning("No processed data path found in config. Skipping save.")
                            return X, y
                            
                        # Create output directory if it doesn't exist
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Save processed data
                        processed_df = pd.concat([X, y], axis=1)
                        processed_df.to_csv(output_path, index=False)
                        logger.info(f"Saved processed data to {output_path}")
                        
                        # Log the processed data as an artifact
                        mlflow.log_artifact(str(output_path))
                    except Exception as e:
                        logger.warning(f"Failed to save processed data: {e}")
                
                return X, y
                
            except Exception as e:
                logger.error(f"Error during preprocessing: {e}")
                raise
            
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scaler."""
        try:
            # First encode categorical features (not training data)
            df = self.encode_categorical_features(df, is_training=False)
            
            # Select features
            X = df[self.feature_columns]
            
            # Separate numeric and categorical features
            numeric_features = [col for col in X.columns if col not in self.categorical_columns]
            categorical_features = [col for col in X.columns if col in self.categorical_columns]
            
            # Handle numeric features
            if numeric_features:
                X_numeric = X[numeric_features]
                X_numeric_imputed = self.imputer.transform(X_numeric)
                X_numeric_scaled = pd.DataFrame(
                    self.scaler.transform(X_numeric_imputed),
                    columns=numeric_features,
                    index=X.index
                )
            else:
                X_numeric_scaled = pd.DataFrame(index=X.index)
            
            # Handle categorical features (already encoded)
            if categorical_features:
                X_categorical = X[categorical_features]
            else:
                X_categorical = pd.DataFrame(index=X.index)
            
            # Combine numeric and categorical features
            return pd.concat([X_numeric_scaled, X_categorical], axis=1)
            
        except KeyError as e:
            raise ValueError(f"Missing required features: {str(e)}")
        except Exception as e:
            logger.error(f"Error transforming new data: {e}")
            raise


def main():
    """Main function to run preprocessing pipeline."""
    try:
        # Get the absolute path to config.yaml
        config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info("Starting preprocessing pipeline...")
        logger.info(f"Using config from: {config_path}")
        
        # Initialize preprocessor
        preprocessor = WeatherDataPreprocessor(config)
        
        # Process the data
        X, y = preprocessor.process()
        
        logger.info(f"Preprocessing completed successfully. Output shape: {X.shape}")
        logger.info(f"Features: {list(X.columns)}")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        raise


if __name__ == "__main__":
    main()