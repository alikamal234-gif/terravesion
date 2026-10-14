import logging
from typing import Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class SeasonalNaiveForecast:
    """
    Predicts future environmental indicator values using seasonal persistence
    (repeating the identical period from the most recent historical annual cycle).
    For MODIS 16-day composites, 1 full year = 23 observations (seasonality=23).
    """
    def __init__(self, seasonality: int = 23):
        self.seasonality = seasonality
        self.history: Optional[pd.Series] = None

    def fit(self, series: pd.Series):
        """
        Fits the baseline model by capturing the historical time series.
        """
        if len(series) < self.seasonality:
            raise ValueError(
                f"Series length ({len(series)}) must be at least the annual seasonal cycle ({self.seasonality})."
            )
        self.history = series.copy()
        logger.info(f"Seasonal Naive baseline fitted on {len(series)} observations (seasonality={self.seasonality}).")

    def predict(self, steps: int = 1, future_index: Optional[pd.DatetimeIndex] = None) -> pd.Series:
        """
        Generates forecasts for `steps` ahead or for the timestamps specified in `future_index`.
        """
        if self.history is None:
            raise ValueError("Model must be fitted before predicting.")

        n_hist = len(self.history)
        predictions = []

        if future_index is not None:
            steps = len(future_index)

        for i in range(steps):
            # Fetch the corresponding seasonal step from the last annual cycle
            # Offset by seasonality
            cycle_offset = self.seasonality - (i % self.seasonality)
            pred_idx = n_hist - cycle_offset
            if pred_idx < 0:
                pred_idx = pred_idx % n_hist
            predictions.append(self.history.iloc[pred_idx])

        if future_index is not None:
            return pd.Series(predictions, index=future_index, name="seasonal_naive")

        if isinstance(self.history.index, pd.DatetimeIndex):
            last_date = self.history.index[-1]
            # 16-day forward step default
            future_idx = pd.date_range(start=last_date + pd.Timedelta(days=16), periods=steps, freq="16D")
            return pd.Series(predictions, index=future_idx, name="seasonal_naive")

        return pd.Series(predictions, name="seasonal_naive")
