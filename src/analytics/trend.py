import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm
from scipy.stats.mstats import theilslopes
import xarray as xr

logger = logging.getLogger(__name__)

def mann_kendall_test(series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Non-parametric Mann-Kendall test for monotonic trend in a time series.
    Tests H0: No monotonic trend against H1: Monotonic increasing/decreasing trend.
    """
    valid = series.dropna()
    n = len(valid)
    if n < 4:
        return {
            "trend": "insufficient_data",
            "tau": float("nan"),
            "p_value": float("nan"),
            "n": n
        }

    t = np.arange(n)
    tau, p_value = kendalltau(t, valid.values)

    trend = (
        "increasing" if tau > 0 and p_value < alpha
        else "decreasing" if tau < 0 and p_value < alpha
        else "no_trend"
    )

    logger.debug(f"Mann-Kendall test complete: {trend} (tau={tau:.4f}, p={p_value:.4e}, n={n})")
    return {
        "trend": trend,
        "tau": float(tau),
        "p_value": float(p_value),
        "n": int(n),
        "is_significant": bool(p_value < alpha)
    }

def calculate_sens_slope(series: pd.Series, confidence: float = 0.95) -> Dict[str, Any]:
    """
    Calculates Sen's slope estimator (Theil-Sen robust linear regression).
    Provides median slope and non-parametric confidence bounds.
    """
    valid = series.dropna()
    n = len(valid)
    if n < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan")
        }

    y = valid.values
    x = np.arange(n)
    slope, intercept, low_slope, up_slope = theilslopes(y, x, confidence)

    # Convert slope per step to slope per year (for 16-day composites: ~22.825 steps/year)
    slope_per_year = slope * 22.825

    return {
        "slope_per_step": float(slope),
        "slope_per_year": float(slope_per_year),
        "intercept": float(intercept),
        "ci_lower": float(low_slope),
        "ci_upper": float(up_slope)
    }

def seasonal_mann_kendall(series: pd.Series, periods_per_year: int = 23, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Seasonal Mann-Kendall test (Hirsch et al. 1982) for seasonal time series.
    Evaluates trends within each seasonal sub-series (16-day composite period)
    and aggregates statistics to eliminate seasonal distortion.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex.")

    df = pd.DataFrame({"val": series})
    doy = df.index.dayofyear
    df["period"] = np.clip((doy - 1) // 16, 0, periods_per_year - 1)

    s_total = 0.0
    var_s_total = 0.0
    valid_periods = 0

    for period_idx, group in df.groupby("period"):
        vals = group["val"].dropna().values
        n_p = len(vals)
        if n_p < 3:
            continue

        # Calculate S for this period: sum_{i < j} sign(x_j - x_i)
        diffs = vals[np.newaxis, :] - vals[:, np.newaxis]
        s_p = np.sum(np.sign(diffs[np.triu_indices(n_p, k=1)]))


        # Variance of S (accounting for ties)
        _, tie_counts = np.unique(vals, return_counts=True)
        tie_term = np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))
        var_s = (n_p * (n_p - 1) * (2 * n_p + 5) - tie_term) / 18.0

        s_total += s_p
        var_s_total += var_s
        valid_periods += 1

    if var_s_total <= 0:
        return mann_kendall_test(series, alpha)

    # Standard normal test statistic Z
    if s_total > 0:
        z = (s_total - 1) / np.sqrt(var_s_total)
    elif s_total < 0:
        z = (s_total + 1) / np.sqrt(var_s_total)
    else:
        z = 0.0

    p_value = 2.0 * (1.0 - norm.cdf(abs(z)))
    trend = (
        "increasing" if z > 0 and p_value < alpha
        else "decreasing" if z < 0 and p_value < alpha
        else "no_trend"
    )

    sen_res = calculate_sens_slope(series)

    return {
        "trend": trend,
        "z_score": float(z),
        "p_value": float(p_value),
        "is_significant": bool(p_value < alpha),
        "valid_seasonal_periods": valid_periods,
        **sen_res
    }

def analyze_trend(series: pd.Series, seasonal: bool = True) -> Dict[str, Any]:
    """
    Runs comprehensive trend test and magnitude estimation.
    """
    if seasonal and isinstance(series.index, pd.DatetimeIndex) and len(series) >= 46:
        return seasonal_mann_kendall(series)
    else:
        mk = mann_kendall_test(series)
        sen = calculate_sens_slope(series)
        return {**mk, **sen}

def compute_spatial_trends(
    da: xr.DataArray,
    step: int = 15,
    variable_name: str = "NDVI"
) -> pd.DataFrame:
    """
    Computes spatial grid trends across Morocco by sampling every `step` pixels
    to prevent memory exhaustion while generating representative spatial trend maps.
    """
    logger.info(f"Computing spatial trend map for {variable_name} (sampling stride={step})...")
    sampled = da.isel(ydim=slice(0, None, step), xdim=slice(0, None, step))
    y_coords = sampled.ydim.values
    x_coords = sampled.xdim.values
    t_coords = pd.to_datetime(sampled.time.values)

    records = []
    data_matrix = sampled.values  # (time, y, x)

    for iy in range(len(y_coords)):
        for ix in range(len(x_coords)):
            ts = data_matrix[:, iy, ix]
            # Skip if mostly NaN (outside Morocco or continuous ocean/cloud)
            valid_mask = ~np.isnan(ts)
            if valid_mask.sum() < 50:
                continue

            s = pd.Series(ts, index=t_coords)
            trend_res = analyze_trend(s, seasonal=True)

            records.append({
                "y": float(y_coords[iy]),
                "x": float(x_coords[ix]),
                "trend": trend_res["trend"],
                "p_value": trend_res["p_value"],
                "slope_per_year": trend_res.get("slope_per_year", float("nan")),
                "is_significant": trend_res.get("is_significant", False)
            })

    result_df = pd.DataFrame(records)
    logger.info(f"Spatial trend computation complete: {len(result_df)} spatial points evaluated.")
    return result_df

class TrendAnalyzer:
    """
    Object-oriented interface for trend analytics.
    """
    def __init__(self, seasonality: int = 23, alpha: float = 0.05):
        self.seasonality = seasonality
        self.alpha = alpha

    def mann_kendall_test(self, series: pd.Series) -> Dict[str, Any]:
        return mann_kendall_test(series, alpha=self.alpha)

    def sen_slope(self, series: pd.Series) -> Dict[str, Any]:
        res = calculate_sens_slope(series)
        return {
            "slope": res["slope_per_step"],
            "slope_per_year": res["slope_per_year"],
            "intercept": res["intercept"],
            "ci_lower": res["ci_lower"],
            "ci_upper": res["ci_upper"]
        }

    def seasonal_mann_kendall(self, series: pd.Series) -> Dict[str, Any]:
        return seasonal_mann_kendall(series, periods_per_year=self.seasonality, alpha=self.alpha)

    def analyze(self, series: pd.Series) -> Dict[str, Any]:
        return analyze_trend(series, seasonal=True)

