import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import logging
import pickle
import os

logger = logging.getLogger(__name__)

class RandomForestForecaster:
    """
    Random Forest Regression model for time-series forecasting.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the Random Forest model on feature matrix X and target y.
        """
        logger.info(f"Training Random Forest with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("Training complete.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generates predictions using the trained model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")
            
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index, name="rf_forecast")
        
    def save(self, filepath: str):
        """
        Saves the fitted model to disk.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model.")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {filepath}")
        
    def load(self, filepath: str):
        """
        Loads a model from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
            
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
