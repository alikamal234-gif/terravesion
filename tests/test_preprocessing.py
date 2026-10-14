import pytest
import numpy as np
import pandas as pd
import xarray as xr
from src.preprocessing.quality import QualityFilter
from src.preprocessing.cleaning import DataCleaner

def test_ndvi_qc_filtering():
    # Synthetic raw NDVI values: -3000 (fill), -4000 (invalid), 5000 (valid, 0.50), 12000 (out of bounds)
    raw_ndvi = xr.DataArray(np.array([-3000, -4000, 5000, 12000], dtype=np.int16))
    qa = xr.DataArray(np.array([0, 0, 0, 0], dtype=np.uint16))
    
    clean = QualityFilter.filter_ndvi_qc(raw_ndvi, qa)
    
    # Only index 2 (5000) should survive as valid
    assert np.isnan(clean.values[0])
    assert np.isnan(clean.values[1])
    assert clean.values[2] == 5000
    assert np.isnan(clean.values[3])

def test_lst_qc_filtering():
    # Synthetic raw LST values: 0 (fill), 65535 (uncalibrated sentinel), 15000 (valid 300K)
    raw_lst = xr.DataArray(np.array([0, 65535, 15000], dtype=np.int32))
    qc = xr.DataArray(np.array([0, 0, 0], dtype=np.uint8))
    
    clean = QualityFilter.filter_lst_qc(raw_lst, qc)
    
    assert np.isnan(clean.values[0])
    assert np.isnan(clean.values[1])
    assert clean.values[2] == 15000

def test_physical_bounds_check():
    valid_da = xr.DataArray(np.array([0.1, 0.4, 0.8]))
    assert QualityFilter.check_physical_bounds(valid_da, min_val=-0.2, max_val=1.0) is True

    violating_da = xr.DataArray(np.array([0.1, 1.5, 0.8]))
    assert QualityFilter.check_physical_bounds(violating_da, min_val=-0.2, max_val=1.0) is False

def test_gap_interpolation_no_forward_invention():
    # Series with a small gap of 1 (should be filled) and a large gap of 4 (should NOT be fully filled)
    series = pd.Series([10.0, np.nan, 30.0, np.nan, np.nan, np.nan, np.nan, 80.0])
    cleaned = DataCleaner.interpolate_time_series(series, max_gap=2)
    
    # Small gap at index 1 is filled with 20.0
    assert cleaned.iloc[1] == 20.0
    # Large gap only limits forward interpolation to max_gap=2, indices 5 and 6 remain NaN
    assert np.isnan(cleaned.iloc[5])
    assert np.isnan(cleaned.iloc[6])
