import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def build_climatological_baseline(series: pd.Series, periods_per_year: int = 23) -> pd.DataFrame:
    """
    Builds a historical multi-year climatological baseline grouped by composite period of year.
    For MODIS 16-day data, periods_per_year=23.
    Uses only historical climatology to ensure NO look-ahead leakage.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex.")

    df = pd.DataFrame({"value": series})
    doy = df.index.dayofyear
    df["period"] = np.clip((doy - 1) // 16, 0, periods_per_year - 1)

    # Compute multi-year mean, std, median, and IQR per seasonal composite period
    climatology = df.groupby("period")["value"].agg(
        mean="mean",
        std="std",
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index()

    climatology["iqr"] = climatology["q75"] - climatology["q25"]

    merged = df.reset_index().merge(climatology, on="period", how="left").set_index(series.index.name or "index")
    merged.index = series.index
    logger.info(f"Built multi-year climatological baseline for {len(series)} observations across {periods_per_year} periods.")
    return merged[["mean", "std", "median", "iqr"]]

def detect_climatological_anomalies(
    series: pd.Series,
    std_multiplier: float = 2.0,
    use_robust: bool = False
) -> pd.DataFrame:
    """
    Detects environmental anomalies by comparing observations against their seasonal climatological baseline.
    If use_robust=True, uses Median +/- (multiplier * 0.7413 * IQR) for outlier robustness.
    """
    clim = build_climatological_baseline(series)
    
    if use_robust:
        baseline = clim["median"]
        dispersion = clim["iqr"] * 0.7413  # Normal-consistent scale from IQR
    else:
        baseline = clim["mean"]
        dispersion = clim["std"]

    upper_bound = baseline + (std_multiplier * dispersion)
    lower_bound = baseline - (std_multiplier * dispersion)
    
    # Anomaly scores (standardized Z-scores)
    z_score = np.where(dispersion > 1e-6, (series - baseline) / dispersion, 0.0)
    is_anomaly = (series > upper_bound) | (series < lower_bound)

    result = pd.DataFrame({
        "actual": series,
        "baseline": baseline,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "z_score": z_score,
        "is_anomaly": is_anomaly
    }, index=series.index)

    anomaly_count = int(is_anomaly.sum())
    logger.info(f"Detected {anomaly_count} climatological anomalies out of {len(series)} points (multiplier={std_multiplier}).")
    return result

def detect_rolling_anomalies(
    series: pd.Series,
    window: int = 5,
    std_multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Detects short-term anomalies using a strictly backward-looking rolling window.
    Strictly center=False to prevent look-ahead data leakage.
    """
    # Shift by 1 so the baseline at time t only uses observations strictly prior to t
    prior_series = series.shift(1)
    rolling_mean = prior_series.rolling(window=window, min_periods=max(2, window // 2)).mean()
    rolling_std = prior_series.rolling(window=window, min_periods=max(2, window // 2)).std()

    upper_bound = rolling_mean + (std_multiplier * rolling_std)
    lower_bound = rolling_mean - (std_multiplier * rolling_std)

    is_anomaly = (series > upper_bound) | (series < lower_bound)

    result = pd.DataFrame({
        "actual": series,
        "baseline": rolling_mean,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "is_anomaly": is_anomaly
    }, index=series.index)

    logger.info(f"Detected {int(is_anomaly.sum())} backward-looking rolling anomalies (window={window}).")
    return result

class AnomalyDetector:
    """
    Object-oriented wrapper for anomaly detection algorithms.
    """
    def __init__(self, seasonality: int = 23):
        self.seasonality = seasonality

    def detect_climatological_anomalies(self, series: pd.Series, threshold_sigma: float = 2.0, use_robust: bool = False) -> pd.DataFrame:
        df = detect_climatological_anomalies(series, std_multiplier=threshold_sigma, use_robust=use_robust)
        df["anomaly_delta"] = df["actual"] - df["baseline"]
        df["climatological_mean"] = df["baseline"]
        return df

    def detect_rolling_anomalies(self, series: pd.Series, window: int = 5, threshold_sigma: float = 2.0) -> pd.DataFrame:
        df = detect_rolling_anomalies(series, window=window, std_multiplier=threshold_sigma)
        df["anomaly_delta"] = df["actual"] - df["baseline"]
        return df

