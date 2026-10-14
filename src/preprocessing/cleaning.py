import logging
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Cleans tabular and raster environmental time series.
    """
    @staticmethod
    def interpolate_time_series(series: pd.Series, max_gap: int = 2, method: str = "linear") -> pd.Series:
        """
        Safely interpolates small gaps in a time series (e.g. cloud gaps <= max_gap),
        preserving NaNs for extended outages without inventing data.
        """
        cleaned = series.interpolate(method=method, limit=max_gap, limit_direction="forward")
        imputed_count = int(series.isnull().sum() - cleaned.isnull().sum())
        logger.info(f"Interpolated {imputed_count} missing values with max gap {max_gap}.")
        return cleaned

    @staticmethod
    def clip_outliers(series: pd.Series, lower_quantile: float = 0.001, upper_quantile: float = 0.999) -> pd.Series:
        """
        Clips extreme sensor noise outliers based on empirical quantiles.
        """
        q_low = series.quantile(lower_quantile)
        q_high = series.quantile(upper_quantile)
        clipped = series.clip(lower=q_low, upper=q_high)
        logger.debug(f"Clipped series between {q_low:.3f} and {q_high:.3f}")
        return clipped
