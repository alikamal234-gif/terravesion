import pandas as pd
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)

class IsolationForestAnomalyDetector:
    """
    Unsupervised ML Anomaly Detection using Isolation Forest.
    """
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination, 
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def train_isolation_forest(self, X: pd.DataFrame):
        """
        Fits the Isolation Forest to the data to learn normal patterns.
        """
        logger.info(f"Training Isolation Forest with {len(X)} samples (contamination: {self.model.contamination}).")
        self.model.fit(X)
        self.is_fitted = True
        logger.info("Training complete.")

    def detect_anomalies(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts whether points in X are anomalies (-1) or normal (1).
        Returns a DataFrame with the original features, anomaly labels, and anomaly scores.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detecting anomalies.")
            
        preds = self.model.predict(X)
        scores = self.model.decision_function(X)
        
        results = X.copy()
        results["is_anomaly"] = (preds == -1)
        results["anomaly_score"] = scores
        
        anomaly_count = results["is_anomaly"].sum()
        logger.info(f"Detected {anomaly_count} anomalies out of {len(X)} samples.")
        
        return results
