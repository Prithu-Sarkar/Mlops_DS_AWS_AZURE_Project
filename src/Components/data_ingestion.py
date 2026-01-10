import os  # File/directory operations and path manipulation
import sys  # System utilities for exception handling
import pandas as pd  # Data manipulation and CSV I/O
from sklearn.model_selection import train_test_split  # Train/test splitting utility

# Local project imports for ML pipeline components
from src.exception import CustomException  # Custom exception with location tracking
from src.logger import logging  # Centralized logging to timestamped files
from dataclasses import dataclass  # Data classes for configuration management

# Pipeline component imports
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion paths in artifacts directory."""
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv') 
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    """Handles complete data ingestion pipeline: read → split → save artifacts."""
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.train_data_path = None
        self.test_data_path = None
    
    def initiate_data_ingestion(self) -> tuple[str, str]:
        """
        Complete data ingestion pipeline:
        1. Read raw CSV data
        2. Create artifacts directory structure
        3. Save raw dataset
        4. Split into train/test (80/20)
        5. Save processed train/test CSVs
        
        Returns:
            tuple[str, str]: Paths to train.csv and test.csv
        """
        logging.info("Entered the data ingestion method or component")
        
        try:
            # Read raw student dataset
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info(f'Read the dataset as dataframe with shape: {df.shape}')
            
            # Create artifacts directory structure
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw dataset first
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f'Raw dataset saved to: {self.ingestion_config.raw_data_path}')
            
            # Perform train/test split (80/20, reproducible)
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
            )
            
            # Save processed datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info(f"Data ingestion completed successfully")
            logging.info(f"Train dataset shape: {train_set.shape}, saved to: {self.ingestion_config.train_data_path}")
            logging.info(f"Test dataset shape: {test_set.shape}, saved to: {self.ingestion_config.test_data_path}")
            
            # Store paths for pipeline continuation
            self.train_data_path = self.ingestion_config.train_data_path
            self.test_data_path = self.ingestion_config.test_data_path
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            # Log and raise detailed error with file/line location
            raise CustomException(e, sys.exc_info())


# Pipeline orchestrator - runs complete ML workflow
if __name__ == "__main__":
    # Step 1: Data Ingestion (creates train.csv, test.csv)
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))