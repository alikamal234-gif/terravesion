import logging
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Mean Absolute Error (MAE).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return float("nan")
    mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
    logger.debug(f"Calculated MAE: {mae:.5f}")
    return mae

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Root Mean Squared Error (RMSE).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return float("nan")
    rmse = float(np.sqrt(np.mean(np.square(y_true[mask] - y_pred[mask]))))
    logger.debug(f"Calculated RMSE: {rmse:.5f}")
    return rmse

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Coefficient of Determination (R^2).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return float("nan")
    ss_res = np.sum(np.square(y_true[mask] - y_pred[mask]))
    ss_tot = np.sum(np.square(y_true[mask] - np.mean(y_true[mask])))
    if ss_tot == 0:
        return 0.0
    r2 = float(1.0 - (ss_res / ss_tot))
    return r2

def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluates MAE, RMSE, and R2 simultaneously.
    """
    return {
        "mae": round(calculate_mae(y_true, y_pred), 4),
        "rmse": round(calculate_rmse(y_true, y_pred), 4),
        "r2": round(calculate_r2(y_true, y_pred), 4)
    }
