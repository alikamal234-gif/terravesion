import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)

def create_lag_features(df: pd.DataFrame, target_cols: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Creates lag features for the specified target columns.
    """
    df = df.copy()
    for col in target_cols:
        for lag in lags:
            new_col = f"{col}_lag_{lag}"
            df[new_col] = df[col].shift(lag)
            logger.debug(f"Created lag feature: {new_col}")
    
    logger.info(f"Created lag features for columns: {target_cols} with lags: {lags}")
    return df
