import os  # Directory creation and path management
import sys  # Exception traceback utilities
import pickle  # Standard Python object serialization
import numpy as np  # Numerical arrays (type hints)
from typing import Dict  # Type hints for model evaluation return

# ML evaluation and optimization
from sklearn.metrics import r2_score  # R² regression metric
from sklearn.model_selection import GridSearchCV  # Hyperparameter search

# Project infrastructure
from src.exception import CustomException  # Enhanced exception handling
from src.logger import logging  # Centralized logging to timestamped files


def save_object(file_path: str, obj: object) -> None:
    """
    Production-ready object serialization for ML artifacts.
    
    Auto-creates directories. Used for models, preprocessors, scalers.
    
    Args:
        file_path (str): Target path e.g., 'artifacts/model.pkl'
        obj (object): Picklable Python object (sklearn model, ColumnTransformer, etc.)
    
    Example:
        save_object('artifacts/preprocessor.pkl', preprocessing_obj)
    """
    try:
        # Ensure artifacts directory structure exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Saving object to: {file_path}")
        
        # High-performance pickle protocol (Python 3.8+ optimized)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
            
        logging.info(f"Object saved successfully: {file_path}")
        
    except Exception as e:
        logging.error(f"Failed to save object {file_path}: {str(e)}")
        raise CustomException(e, sys.exc_info())


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray, 
    y_test: np.ndarray,
    models: Dict[str, object],
    param: Dict[str, dict]
) -> Dict[str, float]:
    """
    Automated model comparison with optional hyperparameter tuning.
    
    Workflow per model:
    1. GridSearchCV (if params provided) → best hyperparameters
    2. Retrain on full training data
    3. Test R² evaluation (leakage-free)
    
    Args:
        X_train/y_train: Training features/target arrays
        X_test/y_test: Holdout test features/target  
        models: Dict of {'ModelName': sklearn_model_instance}
        param: Dict of {'ModelName': hyperparameter_grid}
        
    Returns:
        Dict[str, float]: {'ModelName': test_r2_score}
        
    Example:
        models = {'RF': RandomForestRegressor()}
        params = {'RF': {'n_estimators': [100, 200]}}
        report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
    """
    try:
        report: Dict[str, float] = {}
        
        logging.info(f"Evaluating {len(models)} models...")
        
        for model_name, model in models.items():
            para = param.get(model_name, {})
            
            logging.info(f"Training {model_name}...")
            
            # Hyperparameter optimization (only if params provided)
            if para:
                logging.info(f"Hyperparameter tuning for {model_name}: {len(para)} params")
                gs = GridSearchCV(
                    model, para, cv=3, scoring="r2", 
                    n_jobs=-1, verbose=0  # Parallel + silent
                )
                gs.fit(X_train, y_train)
                best_params = gs.best_params_
                logging.info(f"Best params for {model_name}: {best_params}")
                model.set_params(**best_params)
            else:
                logging.info(f"No hyperparameter tuning for {model_name}")
            
            # Train final model
            model.fit(X_train, y_train)
            
            # Leakage-free test evaluation
            y_test_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Comprehensive logging
            logging.info(f"{model_name} → Test R²: {test_r2:.4f}")
            
            report[model_name] = test_r2
        
        # Summary
        best_model = max(report, key=report.get)
        best_score = report[best_model]
        logging.info(f"Model evaluation complete. Best: {best_model} (R²={best_score:.4f})")
        
        return report
        
    except Exception as e:
        logging.error(f"Model evaluation failed: {str(e)}")
        raise CustomException(e, sys.exc_info())


def load_object(file_path: str) -> object:
    """
    Deserialize ML artifacts for inference/prediction serving.
    
    Args:
        file_path (str): Path to pickled artifact
        
    Returns:
        object: Loaded sklearn model or preprocessor
        
    Example:
        model = load_object('artifacts/model.pkl')
        preprocessor = load_object('artifacts/preprocessor.pkl')
        predictions = model.predict(preprocessor.transform(new_data))
    """
    try:
        logging.info(f"Loading object from: {file_path}")
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Object loaded successfully: {file_path}")
        return obj
        
    except Exception as e:
        logging.error(f"Failed to load object {file_path}: {str(e)}")
        raise CustomException(e, sys.exc_info())


# Production Pipeline Integration Example:
"""
# 1. Training phase
save_object('artifacts/model.pkl', best_model)
save_object('artifacts/preprocessor.pkl', preprocessor)

# 2. Evaluation phase  
models = {'RF': RandomForestRegressor(), 'XGB': XGBRegressor()}
report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

# 3. Inference phase (model serving)
model = load_object('artifacts/model.pkl')
preprocessor = load_object('artifacts/preprocessor.pkl')
new_predictions = model.predict(preprocessor.transform(new_customer_data))
"""
