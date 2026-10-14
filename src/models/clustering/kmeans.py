import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class EcosystemConditionClustering:
    """
    K-Means clustering pipeline to identify environmental condition profiles
    (e.g., classifying joint NDVI/LST behavior).
    """
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        self.is_fitted = False

    def fit_kmeans(self, X: pd.DataFrame):
        """
        Scales features and fits the K-Means algorithm.
        """
        logger.info(f"Fitting K-Means with {self.n_clusters} clusters on {len(X)} samples.")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        logger.info("K-Means training complete.")

    def predict_clusters(self, X: pd.DataFrame) -> pd.Series:
        """
        Assigns cluster labels to new data.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting clusters.")
            
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        
        logger.debug(f"Assigned {len(labels)} data points to {self.n_clusters} clusters.")
        return pd.Series(labels, index=X.index, name="cluster_id")
        
    def analyze_cluster_profiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the average feature values for each cluster to help interpretation.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to analyze cluster profiles.")
            
        X_with_labels = X.copy()
        X_with_labels["cluster_id"] = self.predict_clusters(X)
        
        profiles = X_with_labels.groupby("cluster_id").mean()
        logger.info("Generated mean profiles for cluster interpretation.")
        return profiles
