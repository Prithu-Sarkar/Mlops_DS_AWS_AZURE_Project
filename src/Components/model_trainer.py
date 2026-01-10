import os
import sys
from dataclasses import dataclass
import numpy as np

# ML models
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Project utilities
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

import logging as py_logging

# Configure logging to show INFO messages in console
py_logging.basicConfig(
    level=py_logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    Trains multiple regression models, evaluates them, selects the best, and saves it.
    CatBoost is trained separately due to sklearn incompatibility.
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.best_model = None
        self.best_model_score = None
        self.best_model_name = None

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        try:
            logging.info("Model training started")

            # Split features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info(f"Train shape: X={X_train.shape}, y={y_train.shape}")
            logging.info(f"Test shape: X={X_test.shape}, y={y_test.shape}")

            # Define sklearn-compatible models
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }

            # Hyperparameter grids
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
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
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate sklearn models
            logging.info("Evaluating sklearn models with hyperparameter tuning")
            model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )
            logging.info(f"Sklearn model evaluation completed: {model_report}")

            # Train CatBoost separately
            logging.info("Training CatBoost Regressor separately")
            cat_model = CatBoostRegressor(
                depth=8, learning_rate=0.05, iterations=50, verbose=False, random_state=42
            )
            cat_model.fit(X_train, y_train)
            cat_r2 = r2_score(y_test, cat_model.predict(X_test))
            logging.info(f"CatBoost R² score: {cat_r2:.4f}")

            # Combine results and select best model
            best_sklearn_score = max(model_report.values())
            best_sklearn_name = list(model_report.keys())[list(model_report.values()).index(best_sklearn_score)]

            if cat_r2 > best_sklearn_score:
                self.best_model = cat_model
                self.best_model_score = cat_r2
                self.best_model_name = "CatBoost Regressor"
            else:
                self.best_model = models[best_sklearn_name]
                self.best_model_score = best_sklearn_score
                self.best_model_name = best_sklearn_name

            logging.info(f"Best model: {self.best_model_name} | R²: {self.best_model_score:.4f}")

            # Quality gate
            if self.best_model_score < 0.6:
                raise CustomException(f"No acceptable model found. Best R²: {self.best_model_score:.4f}")

            # Save best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=self.best_model)
            logging.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")

            return self.best_model_score

        except Exception as e:
            raise CustomException(str(e), sys.exc_info())


if __name__ == "__main__":
    try:
        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformation

        logging.info("MODEL TRAINING PIPELINE STARTED")

        # Data ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # Data transformation
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

        # Model training
        trainer = ModelTrainer()
        r2 = trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"MODEL TRAINING COMPLETED | Final R² Score: {r2:.4f}")

    except Exception as e:
        raise CustomException(str(e), sys.exc_info())
