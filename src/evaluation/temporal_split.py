import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def temporal_train_val_test_split(df: pd.DataFrame, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a chronologically ordered DataFrame into train, validation, and test sets 
    without data leakage.
    """
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("Validation and test ratios must sum to less than 1.0.")
        
    df = df.copy()
    if not df.index.is_monotonic_increasing:
        logger.warning("Index is not monotonic increasing. Sorting by index to enforce chronology.")
        df = df.sort_index()

    n = len(df)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - test_size - val_size
    
    train = df.iloc[:train_size]
    val = df.iloc[train_size:train_size + val_size]
    test = df.iloc[train_size + val_size:]
    
    logger.info(f"Temporal split complete. Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test
