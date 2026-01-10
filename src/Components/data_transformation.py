import sys  # System utilities for exception handling
import numpy as np  # Array operations for feature matrix concatenation
import pandas as pd  # Dataframe I/O and manipulation
import os  # File path operations

# Scikit-learn preprocessing pipeline components
from sklearn.compose import ColumnTransformer  # Combine multiple transformers
from sklearn.impute import SimpleImputer  # Handle missing values
from sklearn.pipeline import Pipeline  # Sequential transformation steps
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Encoding + scaling

# Project components
from src.exception import CustomException  # Custom exception with location tracking
from src.logger import logging  # Centralized logging system
from src.utils import save_object  # Custom object serialization utility
from dataclasses import dataclass  # Configuration data classes


@dataclass
class DataTransformationConfig:
    """Configuration paths for transformation artifacts."""
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    """Complete feature engineering pipeline: imputation → encoding → scaling → serialization."""
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.preprocessing_obj = None
    
    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates production-ready preprocessing pipeline for student performance dataset:
        
        Numerical Pipeline: writing_score, reading_score → impute(median) → scale(z-score)
        Categorical Pipeline: gender, race/ethnicity, parental_education, lunch, test_prep → 
                            impute(mode) → one-hot → scale(sparse)
        
        Returns:
            ColumnTransformer: Fitted preprocessing pipeline
        """
        try:
            # Feature columns for student performance dataset
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            # Numerical pipeline: Handle missing → Standardize (mean=0, std=1)
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Robust to outliers
                    ("scaler", StandardScaler())  # Z-score normalization
                ]
            )
            
            # Categorical pipeline: Impute → One-hot → Scale (sparse matrix)
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Mode imputation
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),  # Production-safe
                    ("scaler", StandardScaler(with_mean=False))  # Sparse matrix scaling
                ]
            )
            
            # Combine pipelines into single ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)  # Fixed typo
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys.exc_info())
    
    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Complete transformation pipeline:
        1. Load train/test CSVs
        2. Separate features/target (math_score)
        3. Fit preprocessing pipeline on TRAIN only
        4. Transform both train/test
        5. Concatenate features + target
        6. Serialize preprocessor for inference
        
        Args:
            train_path: Path to train.csv
            test_path: Path to test.csv
            
        Returns:
            tuple: (train_array, test_array, preprocessor_path)
        """
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            
            # Target column for prediction
            target_column_name = "math_score"
            
            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing dataframes")
            
            # Fit on train, transform both (prevents data leakage)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info(f"Train features shape: {input_feature_train_arr.shape}")
            logging.info(f"Test features shape: {input_feature_test_arr.shape}")
            
            # Concatenate features + target for model training
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Serialize preprocessor for model serving/inference
            logging.info(f"Saving preprocessing object at: {self.data_transformation_config.preprocessor_obj_file_path}")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            # Store for class access
            self.preprocessing_obj = preprocessing_obj
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys.exc_info())
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    print("DATA TRANSFORMATION MODULE STARTED")

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
        train_path,
        test_path
    )

    print("DATA TRANSFORMATION COMPLETED")
    print("Train array shape:", train_arr.shape)
    print("Test array shape:", test_arr.shape)
    print("Preprocessor saved at:", preprocessor_path)
