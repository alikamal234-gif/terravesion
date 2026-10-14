import pytest
import os
import pandas as pd
import numpy as np
from src.models.forecasting.baseline import SeasonalNaiveForecast
from src.models.forecasting.random_forest import RandomForestForecaster
from src.evaluation.metrics import evaluate_forecast, calculate_mae, calculate_rmse, calculate_r2

def test_seasonal_naive_forecast():
    # 46 observations (2 full seasonal cycles of 23 steps)
    dates = pd.date_range("2020-01-01", periods=46, freq="16D")
    t = np.arange(46)
    vals = 0.3 + 0.1 * np.sin(2 * np.pi * t / 23.0)
    series = pd.Series(vals, index=dates)

    baseline = SeasonalNaiveForecast(seasonality=23)
    baseline.fit(series)

    # Predict 23 steps into the future
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=16), periods=23, freq="16D")
    preds = baseline.predict(steps=23, future_index=future_dates)

    assert len(preds) == 23
    # Because series is perfectly periodic with period 23, the future prediction should match the last cycle exactly
    expected_last_cycle = vals[-23:]
    np.testing.assert_allclose(preds.values, expected_last_cycle, atol=1e-5)

def test_random_forest_forecaster_fit_and_save():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "lag1": np.random.uniform(0.1, 0.6, n),
        "lag23": np.random.uniform(0.1, 0.6, n),
        "sin_doy": np.sin(np.linspace(0, 2 * np.pi, n))
    })
    y = 0.5 * X["lag1"] + 0.5 * X["lag23"] + np.random.normal(0, 0.01, n)

    rf = RandomForestForecaster(n_estimators=30, max_depth=5, random_state=42)
    rf.fit(X, y)
    preds = rf.predict(X)

    metrics = evaluate_forecast(y.values, preds.values)
    assert metrics["mae"] < 0.05
    assert metrics["r2"] > 0.85

    # Test save and reload
    ckpt_path = "data/processed/checkpoints/test_rf_model.pkl"
    try:
        rf.save(ckpt_path)
        assert os.path.exists(ckpt_path)

        rf_loaded = RandomForestForecaster()
        rf_loaded.load(ckpt_path)
        loaded_preds = rf_loaded.predict(X)
        np.testing.assert_allclose(preds.values, loaded_preds.values)
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

def test_evaluation_metrics():
    y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    assert calculate_mae(y_true, y_pred) == 0.0
    assert calculate_rmse(y_true, y_pred) == 0.0
    assert calculate_r2(y_true, y_pred) == 1.0

    y_perturbed = y_true + 0.05
    assert calculate_mae(y_true, y_perturbed) == pytest.approx(0.05)
    assert calculate_rmse(y_true, y_perturbed) == pytest.approx(0.05)
