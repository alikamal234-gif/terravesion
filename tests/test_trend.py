import pytest
import pandas as pd
import numpy as np
from src.analytics.trend import mann_kendall_test, calculate_sens_slope, seasonal_mann_kendall, TrendAnalyzer

def test_trend_increasing():
    np.random.seed(42)
    t = np.arange(50)
    # Strictly increasing with subtle noise
    vals = 0.2 + 0.01 * t + np.random.normal(0, 0.001, size=50)
    series = pd.Series(vals)
    
    mk = mann_kendall_test(series)
    sen = calculate_sens_slope(series)
    
    assert mk["trend"] == "increasing"
    assert mk["is_significant"] is True
    assert mk["p_value"] < 0.01
    assert sen["slope_per_step"] > 0

def test_trend_decreasing():
    np.random.seed(42)
    t = np.arange(50)
    # Strictly decreasing with subtle noise
    vals = 0.8 - 0.01 * t + np.random.normal(0, 0.001, size=50)
    series = pd.Series(vals)
    
    mk = mann_kendall_test(series)
    sen = calculate_sens_slope(series)
    
    assert mk["trend"] == "decreasing"
    assert mk["is_significant"] is True
    assert mk["p_value"] < 0.01
    assert sen["slope_per_step"] < 0

def test_trend_flat_no_trend():
    np.random.seed(42)
    # Constant baseline plus pure zero-mean Gaussian noise
    vals = 0.4 + np.random.normal(0, 0.02, size=100)
    series = pd.Series(vals)
    
    mk = mann_kendall_test(series)
    assert mk["trend"] == "no_trend"
    assert mk["p_value"] > 0.05

def test_seasonal_mann_kendall():
    np.random.seed(42)
    # 5 years of 16-day composites (115 steps)
    dates = pd.date_range("2020-01-01", periods=115, freq="16D")
    t = np.arange(115)
    # Seasonal sine cycle + robust long term increase
    vals = 0.3 + 0.1 * np.sin(2 * np.pi * t / 23.0) + 0.005 * t + np.random.normal(0, 0.002, size=115)
    series = pd.Series(vals, index=dates)
    
    analyzer = TrendAnalyzer(seasonality=23)
    smk = analyzer.seasonal_mann_kendall(series)
    
    assert smk["trend"] == "increasing"
    assert smk["is_significant"] is True
    assert smk["slope_per_year"] > 0
