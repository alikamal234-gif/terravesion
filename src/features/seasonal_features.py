import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts basic seasonal/temporal components from a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex to extract seasonal features.")
        
    df = df.copy()
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['season'] = (df.index.month % 12 + 3) // 3  # 1:Winter, 2:Spring, 3:Summer, 4:Fall
    
    logger.info("Created seasonal features (month, quarter, year, season).")
    return df
