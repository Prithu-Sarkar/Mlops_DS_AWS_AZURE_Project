import os  # File path operations for model artifacts
import sys  # System utilities for exception handling
from dataclasses import dataclass  # Configuration data classes
import numpy as np
# Import CatBoost
from catboost import CatBoostClassifier, CatBoostRegressor
import sys
print(sys.executable)

# Import XGBoost
from xgboost import XGBClassifier, XGBRegressor
# Model candidates for automated evaluation
from catboost import CatBoostRegressor  # Gradient boosting with categorical support
from sklearn.ensemble import (
    AdaBoostRegressor,      # Boosting with weak learners
    GradientBoostingRegressor,  # Tree boosting
    RandomForestRegressor,  # Bagging ensemble
)
from sklearn.linear_model import LinearRegression  # Baseline linear model
from sklearn.metrics import r2_score  # R² score for regression evaluation
from sklearn.neighbors import KNeighborsRegressor  # Instance-based learning
from sklearn.tree import DecisionTreeRegressor  # Single decision tree
from xgboost import XGBRegressor  # Extreme gradient boosting

# Project components
from src.exception import CustomException  # Custom exception with location tracking
from src.logger import logging  # Centralized logging system
from src.utils import save_object, evaluate_models  # Utility functions


@dataclass
class ModelTrainerConfig:
    """Path configuration for trained model serialization."""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    Automated model selection and training:
    1. Evaluates 7 regression algorithms with hyperparameter grids
    2. Selects best model by R² score (train + test)
    3. Serializes best model for inference
    4. Returns final test R² score
    """
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.best_model = None
        self.best_model_score = None
        self.best_model_name = None
    
    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        """
        Complete model training pipeline with automated hyperparameter tuning.
        
        Args:
            train_array: NumPy array [features | target] from data transformation
            test_array: NumPy array [features | target] from data transformation
            
        Returns:
            float: R² score of best model on test set
        """
        try:
            logging.info("Model training started")
            
            # Extract features and target (last column is target)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            logging.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
            logging.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
            
            # Model zoo with default hyperparameters
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(random_state=42),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }
            
            # Hyperparameter grids for automated tuning (via evaluate_models utility)
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }
            
            # Evaluate all models with hyperparameter tuning
            logging.info("Initiating model evaluation and hyperparameter tuning")
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )
            
            logging.info(f"Model evaluation completed. Report: {model_report}")
            
            # Find best model by R² score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            # Store best model details
            self.best_model = best_model
            self.best_model_score = best_model_score
            self.best_model_name = best_model_name
            
            logging.info(f"Best model found: {best_model_name} with R² score: {best_model_score:.4f}")
            
            # Quality gate: Reject poor models
            if best_model_score < 0.6:
                raise CustomException(f"No acceptable model found. Best R²: {best_model_score:.4f}")
            
            logging.info("Best model meets quality threshold. Saving to artifacts...")
            
            # Serialize best model for inference
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Final test evaluation
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            logging.info(f"Final test R² score: {r2_square:.4f}")
            logging.info(f"Model saved successfully at: {self.model_trainer_config.trained_model_file_path}")
            
            return r2_square
            
        except Exception as e:
            raise CustomException(e, sys.exc_info())
