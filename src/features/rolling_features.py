import logging
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)

def create_rolling_features(
    df: pd.DataFrame,
    target_cols: List[str],
    windows: List[int],
    agg_func: str = "mean",
    shift: int = 1
) -> pd.DataFrame:
    """
    Creates rolling window features for the specified target columns.
    Applies shift=1 by default to ensure only strictly prior observations are used,
    preventing look-ahead or contemporaneous target leakage.
    """
    df = df.copy()
    for col in target_cols:
        source_series = df[col].shift(shift) if shift > 0 else df[col]
        for window in windows:
            new_col = f"{col}_rolling_{agg_func}_{window}"
            if agg_func == "mean":
                df[new_col] = source_series.rolling(window=window, min_periods=1).mean()
            elif agg_func == "std":
                df[new_col] = source_series.rolling(window=window, min_periods=1).std()
            elif agg_func == "min":
                df[new_col] = source_series.rolling(window=window, min_periods=1).min()
            elif agg_func == "max":
                df[new_col] = source_series.rolling(window=window, min_periods=1).max()
            else:
                raise ValueError(f"Unsupported aggregation function: {agg_func}")
            
            logger.debug(f"Created rolling feature: {new_col} (shift={shift})")
            
    logger.info(f"Created rolling features for columns: {target_cols} with windows: {windows} (shift={shift})")
    return df
