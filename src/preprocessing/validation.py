import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates data schemas and checks for basic quality issues.
    """
    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Checks for missing values and logs a report.
        """
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values detected:\n{missing[missing > 0]}")
        else:
            logger.info("No missing values detected.")
        return df
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
        """
        Checks for duplicate rows.
        """
        duplicates = df.duplicated(subset=subset).sum()
        if duplicates > 0:
            logger.warning(f"{duplicates} duplicate rows detected.")
        else:
            logger.info("No duplicate rows detected.")
        return df

    @staticmethod
    def validate_schema(df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """
        Validates that the DataFrame contains expected columns.
        """
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Schema validation failed. Missing columns: {missing_cols}")
            return False
        logger.info("Schema validation passed.")
        return True
