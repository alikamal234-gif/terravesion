import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts cyclic, seasonal, and trend temporal features from a DatetimeIndex
    tailored to NASA 16-day composite cycles (~23 periods per annual cycle).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex to extract temporal features.")

    df = df.copy()
    doy = df.index.dayofyear
    
    # 1. Day of Year cyclical encoding
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    
    # 2. 16-day composite period of year index (0 to 22)
    df["period_16d"] = (doy - 1) // 16
    df["sin_period_16d"] = np.sin(2 * np.pi * df["period_16d"] / 23.0)
    df["cos_period_16d"] = np.cos(2 * np.pi * df["period_16d"] / 23.0)
    
    # 3. Calendar components
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["year_trend"] = df.index.year - df.index.year.min()
    df["season"] = (df.index.month % 12 + 3) // 3  # 1:Winter, 2:Spring, 3:Summer, 4:Fall

    logger.info("Created cyclical, period_16d, and seasonal temporal features.")
    return df
