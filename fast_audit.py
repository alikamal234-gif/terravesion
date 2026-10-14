import os
import json
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta

ndvi_file = r'c:\Users\Youcode\Desktop\Terravesion AI\data\raw\ndvi\MOD13Q1.061_250m_aid0001.nc'
lst_file = r'c:\Users\Youcode\Desktop\Terravesion AI\data\raw\lst\MOD11A2.061_1km_aid0001.nc'

audit = {'NDVI': {}, 'LST': {}}

def process_ds(ds, dataset_key, filepath):
    audit[dataset_key]['file_name'] = os.path.basename(filepath)
    
    # 1. Temporal Coverage
    t_var = ds.variables.get('time')
    if t_var is not None:
        times = t_var[:]
        units = t_var.units if hasattr(t_var, 'units') else 'unknown'
        calendar = t_var.calendar if hasattr(t_var, 'calendar') else 'standard'
        audit[dataset_key]['temporal'] = {
            'first_timestamp': float(times[0]),
            'last_timestamp': float(times[-1]),
            'time_units': units,
            'calendar': calendar,
            'count': len(times)
        }
        
        # Check gaps
        if len(times) > 1:
            diffs = np.diff(times)
            expected_step = np.median(diffs)
            gaps = int(np.sum(diffs > expected_step * 1.5))
            audit[dataset_key]['temporal']['expected_step'] = float(expected_step)
            audit[dataset_key]['temporal']['missing_steps'] = gaps
            audit[dataset_key]['temporal']['is_regular'] = (gaps == 0)

    # 2. Spatial Coverage
    x_var = ds.variables.get('xdim') or ds.variables.get('lon')
    y_var = ds.variables.get('ydim') or ds.variables.get('lat')
    
    if x_var is not None and y_var is not None:
        x_data = x_var[:]
        y_data = y_var[:]
        x_res = abs(x_data[1] - x_data[0]) if len(x_data) > 1 else 0
        y_res = abs(y_data[1] - y_data[0]) if len(y_data) > 1 else 0
        
        crs_var = ds.variables.get('crs')
        crs_info = {attr: crs_var.getncattr(attr) for attr in crs_var.ncattrs()} if crs_var else {}
        
        audit[dataset_key]['spatial'] = {
            'x_min': float(x_data.min()),
            'x_max': float(x_data.max()),
            'y_min': float(y_data.min()),
            'y_max': float(y_data.max()),
            'x_resolution': float(x_res),
            'y_resolution': float(y_res),
            'coordinate_names': [x_var.name, y_var.name],
            'crs_metadata': crs_info
        }

    # 3. Global Attributes
    audit[dataset_key]['global_attributes'] = {attr: str(ds.getncattr(attr)) for attr in ds.ncattrs()}
    
    # 4 & 5/6. Main Variable Quality
    main_var_name = None
    for v in ds.variables:
        if 'NDVI' in v.upper() or ('LST' in v.upper() and 'DAY' in v.upper()):
            main_var_name = v
            break
            
    if main_var_name:
        var = ds.variables[main_var_name]
        audit[dataset_key]['variable_attributes'] = {attr: str(var.getncattr(attr)) for attr in var.ncattrs()}
        
        scale = getattr(var, 'scale_factor', 1.0)
        offset = getattr(var, 'add_offset', 0.0)
        fill_val = getattr(var, '_FillValue', None)
        
        # Load ONE slice for fast memory processing
        slice_data = var[0, :, :]
        
        # Determine invalid mask
        if dataset_key == 'NDVI':
            invalid_mask = (slice_data == fill_val) | (slice_data < -2000)
        else: # LST
            invalid_mask = (slice_data == fill_val) | (slice_data == 65535) | (slice_data < 7500)
            
        nodata_pixels = int(np.sum(invalid_mask))
        total_pixels = slice_data.size
        
        valid_data = slice_data[~invalid_mask]
        
        if len(valid_data) > 0:
            scaled_valid = (valid_data * scale) + offset
            
            stats = {
                'total_pixels_per_slice': total_pixels,
                'invalid_pixels_per_slice': nodata_pixels,
                'invalid_percentage': float((nodata_pixels / total_pixels) * 100),
                'valid_min': float(scaled_valid.min()),
                'valid_max': float(scaled_valid.max()),
                'valid_mean': float(scaled_valid.mean())
            }
            
            if dataset_key == 'LST':
                # Convert Kelvin to Celsius
                stats['valid_min_celsius'] = stats['valid_min'] - 273.15
                stats['valid_max_celsius'] = stats['valid_max'] - 273.15
                stats['valid_mean_celsius'] = stats['valid_mean'] - 273.15
                
            audit[dataset_key]['quality_sample'] = stats
        
        # 11. Computational
        bytes_per_slice = slice_data.nbytes
        total_slices = len(t_var[:]) if t_var else 1
        audit[dataset_key]['computational'] = {
            'array_shape': [total_slices] + list(slice_data.shape),
            'raw_memory_mb': float(bytes_per_slice * total_slices) / (1024**2)
        }

try:
    ds_ndvi = nc.Dataset(ndvi_file, 'r')
    process_ds(ds_ndvi, 'NDVI', ndvi_file)
    ds_ndvi.close()
except Exception as e:
    audit['NDVI']['error'] = str(e)

try:
    ds_lst = nc.Dataset(lst_file, 'r')
    process_ds(ds_lst, 'LST', lst_file)
    ds_lst.close()
except Exception as e:
    audit['LST']['error'] = str(e)

out_json = r'c:\Users\Youcode\Desktop\Terravesion AI\data\audit_ndvi_lst.json'
with open(out_json, 'w') as f:
    json.dump(audit, f, indent=4)
print(f"JSON audit saved to {out_json}")
