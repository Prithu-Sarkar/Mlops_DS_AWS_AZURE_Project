import os  # Directory creation and path manipulation
import sys  # System utilities for exception handling
import numpy as np  # Array operations (unused but imported for consistency)
import pandas as pd  # Dataframe operations (unused but imported for consistency)
import pickle  # Python object serialization (standard library)

# ML evaluation utilities
from sklearn.metrics import r2_score  # R² regression metric
from sklearn.model_selection import GridSearchCV  # Hyperparameter optimization

# Project components
from src.exception import CustomException  # Custom exception with location tracking


def save_object(file_path: str, obj: object) -> None:
    """
    Serializes Python objects (models, preprocessors) to disk using pickle.
    
    Creates parent directories automatically. Handles ML pipeline artifacts.
    
    Args:
        file_path (str): Destination path (e.g., 'artifacts/model.pkl')
        obj (object): Model, preprocessor, or any picklable object
    """
    try:
        # Ensure parent directories exist (artifacts/, models/, etc.)
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Serialize with protocol 4 (Python 3.4+ optimized)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys.exc_info())


def evaluate_models(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    models: dict, 
    param: dict
) -> dict[str, float]:
    """
    Automated model evaluation + hyperparameter tuning across multiple algorithms.
    
    For each model:
    1. GridSearchCV finds best hyperparameters (3-fold CV on train)
    2. Retrains with best params on full train set
    3. Evaluates R² on test set (no overfitting bias)
    
    Args:
        X_train, y_train: Training features/target
        X_test, y_test: Test features/target  
        models (dict): {'ModelName': model_instance, ...}
        param (dict): {'ModelName': {hyperparam_grid}, ...}
        
    Returns:
        dict: {'ModelName': test_r2_score, ...}
    """
    try:
        report = {}
        
        # Iterate through model zoo
        for model_name in models.keys():
            model = models[model_name]
            para = param.get(model_name, {})  # Handle models without params
            
            logging.info(f"Training {model_name} with GridSearchCV...")
            
            # Hyperparameter optimization (3-fold CV prevents overfitting)
            gs = GridSearchCV(model, para, cv=3, scoring='r2', n_jobs=-1)
            gs.fit(X_train, y_train)
            
            # Use best hyperparameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # R² scores (test score = key metric)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Log performance
            logging.info(f"{model_name} → Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            
            # Report test score only (production metric)
            report[model_name] = test_r2
            
        logging.info(f"Model evaluation complete. Best test R²: {max(report.values()):.4f}")
        return report
        
    except Exception as e:
        raise CustomException(e, sys.exc_info())


def load_object(file_path: str) -> object:
    """
    Deserializes pickled ML artifacts (models, preprocessors) for inference.
    
    Args:
        file_path (str): Path to pickled file (e.g., 'artifacts/model.pkl')
        
    Returns:
        object: Loaded model/preprocessor instance
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
            
    except Exception as e:
        raise CustomException(e, sys.exc_info())


# Usage Examples:
"""
# Save trained model
save_object('artifacts/model.pkl', best_model)

# Evaluate 7 models automatically  
model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

# Load for prediction
model = load_object('artifacts/model.pkl')
predictions = model.predict(new_data)
"""
