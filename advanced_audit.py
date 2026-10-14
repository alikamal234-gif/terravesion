import os
import json
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime

ndvi_file = r'c:\Users\Youcode\Desktop\Terravesion AI\data\raw\ndvi\MOD13Q1.061_250m_aid0001.nc'
lst_file = r'c:\Users\Youcode\Desktop\Terravesion AI\data\raw\lst\MOD11A2.061_1km_aid0001.nc'

audit = {'NDVI': {}, 'LST': {}}

def dt64_to_str(dt64):
    if isinstance(dt64, np.datetime64):
        # Handle nanoseconds issue
        ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        return datetime.utcfromtimestamp(ts).isoformat()
    return str(dt64)

# ----------------- NDVI Audit -----------------
try:
    ds_ndvi = xr.open_dataset(ndvi_file)
    audit['NDVI']['global_attributes'] = dict(ds_ndvi.attrs)
    
    # Temporal
    times = ds_ndvi['time'].values
    time_attrs = ds_ndvi['time'].attrs
    audit['NDVI']['temporal'] = {
        'first_timestamp': dt64_to_str(times[0]) if np.issubdtype(times.dtype, np.datetime64) else str(times[0]),
        'last_timestamp': dt64_to_str(times[-1]) if np.issubdtype(times.dtype, np.datetime64) else str(times[-1]),
        'metadata': time_attrs,
        'count': len(times)
    }
    
    # Check regular timestamps (16-day)
    if np.issubdtype(times.dtype, np.datetime64):
        diffs = np.diff(times).astype('timedelta64[D]')
        expected_diff = np.timedelta64(16, 'D')
        missing_steps = len(times) - 1 - (times[-1] - times[0]).astype('timedelta64[D]') / expected_diff
        audit['NDVI']['temporal']['is_regular'] = float(np.mean(diffs)) == 16.0
        audit['NDVI']['temporal']['avg_step_days'] = float(np.mean(diffs))
    
    # Spatial
    if 'xdim' in ds_ndvi.dims and 'ydim' in ds_ndvi.dims:
        xdim, ydim = 'xdim', 'ydim'
    elif 'lon' in ds_ndvi.dims and 'lat' in ds_ndvi.dims:
        xdim, ydim = 'lon', 'lat'
        
    audit['NDVI']['spatial'] = {
        'x_min': float(ds_ndvi[xdim].min()),
        'x_max': float(ds_ndvi[xdim].max()),
        'y_min': float(ds_ndvi[ydim].min()),
        'y_max': float(ds_ndvi[ydim].max()),
        'x_resolution': float(ds_ndvi[xdim].diff(xdim).mean()),
        'y_resolution': float(ds_ndvi[ydim].diff(ydim).mean()),
        'crs_info': dict(ds_ndvi['crs'].attrs) if 'crs' in ds_ndvi else {}
    }
    
    # Main Variable Quality (sample first 3 slices to avoid memory issues)
    main_var = '_250m_16_days_NDVI'
    da = ds_ndvi[main_var]
    audit['NDVI']['variable_attributes'] = dict(da.attrs)
    
    fill_val = da.attrs.get('_FillValue', -3000)
    scale = da.attrs.get('scale_factor', 0.0001)
    offset = da.attrs.get('add_offset', 0.0)
    
    # Sample first time slice
    sample_data = da.isel(time=0).values.astype(float)
    total_pixels = sample_data.size
    nodata_pixels = np.sum(sample_data == fill_val)
    
    valid_data = sample_data[sample_data != fill_val]
    if len(valid_data) > 0:
        scaled_valid = (valid_data * scale) + offset
        audit['NDVI']['quality_sample'] = {
            'total_pixels_per_slice': total_pixels,
            'nodata_pixels_per_slice': int(nodata_pixels),
            'nodata_percentage': float((nodata_pixels / total_pixels) * 100),
            'valid_min_scaled': float(scaled_valid.min()),
            'valid_max_scaled': float(scaled_valid.max()),
            'valid_mean_scaled': float(scaled_valid.mean())
        }
    
    # Memory estimation
    # 271 * 7224 * 7206 * 2 bytes = ~28 GB as float64, ~4 GB as int16
    audit['NDVI']['computational'] = {
        'raw_array_size_mb': (da.size * da.dtype.itemsize) / (1024**2),
        'float64_array_size_mb': (da.size * 8) / (1024**2)
    }
    
    ds_ndvi.close()
except Exception as e:
    audit['NDVI']['error'] = str(e)


# ----------------- LST Audit -----------------
try:
    ds_lst = xr.open_dataset(lst_file)
    audit['LST']['global_attributes'] = dict(ds_lst.attrs)
    
    # Temporal
    times = ds_lst['time'].values
    time_attrs = ds_lst['time'].attrs
    audit['LST']['temporal'] = {
        'first_timestamp': dt64_to_str(times[0]) if np.issubdtype(times.dtype, np.datetime64) else str(times[0]),
        'last_timestamp': dt64_to_str(times[-1]) if np.issubdtype(times.dtype, np.datetime64) else str(times[-1]),
        'metadata': time_attrs,
        'count': len(times)
    }
    
    if np.issubdtype(times.dtype, np.datetime64):
        diffs = np.diff(times).astype('timedelta64[D]')
        audit['LST']['temporal']['avg_step_days'] = float(np.mean(diffs))
    
    # Spatial
    if 'xdim' in ds_lst.dims and 'ydim' in ds_lst.dims:
        xdim, ydim = 'xdim', 'ydim'
    elif 'lon' in ds_lst.dims and 'lat' in ds_lst.dims:
        xdim, ydim = 'lon', 'lat'
        
    audit['LST']['spatial'] = {
        'x_min': float(ds_lst[xdim].min()),
        'x_max': float(ds_lst[xdim].max()),
        'y_min': float(ds_lst[ydim].min()),
        'y_max': float(ds_lst[ydim].max()),
        'x_resolution': float(ds_lst[xdim].diff(xdim).mean()),
        'y_resolution': float(ds_lst[ydim].diff(ydim).mean()),
        'crs_info': dict(ds_lst['crs'].attrs) if 'crs' in ds_lst else {}
    }
    
    # Main Variable Quality
    main_var = 'LST_Day_1km'
    da = ds_lst[main_var]
    audit['LST']['variable_attributes'] = dict(da.attrs)
    
    fill_val = da.attrs.get('_FillValue', 0)
    scale = da.attrs.get('scale_factor', 0.02)
    offset = da.attrs.get('add_offset', 0.0)
    
    # Identify if 65535 is a legitimate value or error code. Raw 65535 * 0.02 = 1310.7 Kelvin which is physically impossible. 
    # Therefore, 65535 is definitely a missing/error flag not captured in _FillValue properly or representing something else.
    
    sample_data = da.isel(time=0).values.astype(float)
    total_pixels = sample_data.size
    
    # Treat 0 and 65535 as invalid for stats
    invalid_mask = (sample_data == fill_val) | (sample_data == 65535) | (sample_data < 7500)
    nodata_pixels = np.sum(invalid_mask)
    
    valid_data = sample_data[~invalid_mask]
    if len(valid_data) > 0:
        scaled_valid_k = (valid_data * scale) + offset
        scaled_valid_c = scaled_valid_k - 273.15
        audit['LST']['quality_sample'] = {
            'total_pixels_per_slice': total_pixels,
            'invalid_pixels_per_slice': int(nodata_pixels),
            'invalid_percentage': float((nodata_pixels / total_pixels) * 100),
            'valid_min_celsius': float(scaled_valid_c.min()),
            'valid_max_celsius': float(scaled_valid_c.max()),
            'valid_mean_celsius': float(scaled_valid_c.mean())
        }
        
    audit['LST']['computational'] = {
        'raw_array_size_mb': (da.size * da.dtype.itemsize) / (1024**2),
        'float64_array_size_mb': (da.size * 8) / (1024**2)
    }
    
    ds_lst.close()
except Exception as e:
    audit['LST']['error'] = str(e)

# Output
out_json = r'c:\Users\Youcode\Desktop\Terravesion AI\data\audit_ndvi_lst.json'
with open(out_json, 'w') as f:
    json.dump(audit, f, indent=4)
print(f"JSON audit saved to {out_json}")
