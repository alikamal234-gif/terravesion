import pytest
import pandas as pd
import numpy as np
from src.features.temporal_features import create_temporal_features
from src.features.lag_features import create_lag_features
from src.features.rolling_features import create_rolling_features

def test_temporal_features():
    # 23 observations representing 1 full year of 16-day composites
    dates = pd.date_range("2020-01-01", periods=23, freq="16D")
    df = pd.DataFrame({"value": np.linspace(0.2, 0.6, 23)}, index=dates)
    
    feat_df = create_temporal_features(df)
    
    assert "sin_doy" in feat_df.columns
    assert "cos_doy" in feat_df.columns
    assert "period_16d" in feat_df.columns
    assert "sin_period_16d" in feat_df.columns
    assert "cos_period_16d" in feat_df.columns
    
    # Cyclical bounds check
    assert feat_df["sin_doy"].min() >= -1.0
    assert feat_df["sin_doy"].max() <= 1.0
    assert feat_df["period_16d"].min() >= 0
    assert feat_df["period_16d"].max() <= 22

def test_lag_features():
    df = pd.DataFrame({"ndvi": [0.1, 0.2, 0.3, 0.4, 0.5]})
    lag_df = create_lag_features(df, target_cols=["ndvi"], lags=[1, 2])
    
    assert "ndvi_lag_1" in lag_df.columns
    assert "ndvi_lag_2" in lag_df.columns
    assert np.isnan(lag_df.loc[0, "ndvi_lag_1"])
    assert lag_df.loc[1, "ndvi_lag_1"] == 0.1
    assert lag_df.loc[2, "ndvi_lag_2"] == 0.1

def test_rolling_features_no_lookahead_leakage():
    # Verify that rolling mean with shift=1 does NOT contain contemporaneous value
    df = pd.DataFrame({"ndvi": [10.0, 20.0, 30.0, 40.0, 50.0]})
    roll_df = create_rolling_features(df, target_cols=["ndvi"], windows=[2], agg_func="mean", shift=1)
    
    col = "ndvi_rolling_mean_2"
    assert col in roll_df.columns
    
    # At index 0, shifted value is NaN -> rolling mean is NaN
    assert np.isnan(roll_df.loc[0, col])
    # At index 1, shifted values available: [10.0] -> mean is 10.0 (NOT (10+20)/2 = 15)
    assert roll_df.loc[1, col] == 10.0
    # At index 2, shifted values available: [10.0, 20.0] -> mean is 15.0 (NOT (20+30)/2 = 25)
    assert roll_df.loc[2, col] == 15.0
