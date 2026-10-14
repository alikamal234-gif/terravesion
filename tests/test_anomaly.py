import pytest
import pandas as pd
import numpy as np
from src.analytics.anomaly import build_climatological_baseline, detect_climatological_anomalies, detect_rolling_anomalies, AnomalyDetector

def test_climatological_baseline():
    # 2 years of 16-day composites (46 steps)
    dates = pd.date_range("2020-01-01", periods=46, freq="16D")
    t = np.arange(46)
    # Seasonal values repeated across the two years
    vals = 0.3 + 0.1 * np.sin(2 * np.pi * t / 23.0)
    series = pd.Series(vals, index=dates)

    baseline = build_climatological_baseline(series, periods_per_year=23)
    assert len(baseline) == 46
    assert "mean" in baseline.columns
    assert "std" in baseline.columns
    # Because year 1 and year 2 have identical seasonal points, std should be virtually 0
    assert baseline["std"].max() < 1e-4

def test_climatological_anomaly_spike_detection():
    # 10 years of 16-day composites (230 steps, matching decadal NASA record length)
    dates = pd.date_range("2015-01-01", periods=230, freq="16D")
    t = np.arange(230)
    np.random.seed(42)
    vals = 0.3 + 0.1 * np.sin(2 * np.pi * t / 23.0) + np.random.normal(0, 0.005, size=230)
    # Inject severe negative drought anomaly (-0.30 drop at step 50)
    vals[50] -= 0.30
    series = pd.Series(vals, index=dates)

    detector = AnomalyDetector(seasonality=23)
    res = detector.detect_climatological_anomalies(series, threshold_sigma=2.0)

    assert bool(res.loc[dates[50], "is_anomaly"]) is True
    assert res.loc[dates[50], "z_score"] < -2.0
    # Other undisturbed points should not be flagged as severe anomalies
    assert bool(res.loc[dates[10], "is_anomaly"]) is False



def test_rolling_anomaly_no_future_leakage():
    # Verify that anomaly at time t does NOT influence baseline at time t
    dates = pd.date_range("2020-01-01", periods=10, freq="16D")
    # Constant series with sudden spike at step 5
    vals = [0.2, 0.2, 0.2, 0.2, 0.2, 0.9, 0.2, 0.2, 0.2, 0.2]
    series = pd.Series(vals, index=dates)

    detector = AnomalyDetector()
    res = detector.detect_rolling_anomalies(series, window=3, threshold_sigma=2.0)

    # Step 5 is the spike (0.9), its baseline is built only from prior steps [0.2, 0.2, 0.2] = 0.2
    assert res.loc[dates[5], "baseline"] == pytest.approx(0.2, abs=1e-4)
    assert res.loc[dates[5], "is_anomaly"] == True

def test_isolation_forest_anomaly_detection():
    from src.models.anomaly.isolation_forest import IsolationForestAnomalyDetector
    np.random.seed(42)
    # 100 normal observations + 5 extreme outliers
    normal = np.random.normal(loc=0.2, scale=0.02, size=(100, 2))
    outliers = np.array([[0.9, 0.9], [0.85, 0.95], [0.92, 0.88], [0.88, 0.91], [0.95, 0.89]])
    data = np.vstack([normal, outliers])
    df = pd.DataFrame(data, columns=["ndvi", "lst"])

    detector = IsolationForestAnomalyDetector(contamination=0.05, random_state=42)
    detector.train_isolation_forest(df)
    results = detector.detect_anomalies(df)

    assert "is_anomaly" in results.columns
    assert "anomaly_score" in results.columns
    assert results["is_anomaly"].sum() > 0
    # Extreme outlier at the end should be detected
    assert bool(results.iloc[-1]["is_anomaly"]) is True
