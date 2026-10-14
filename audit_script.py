import os
import json
import numpy as np
import xarray as xr

ndvi_file = r'c:\Users\Youcode\Desktop\Terravesion AI\data\raw\ndvi\MOD13Q1.061_250m_aid0001.nc'
lst_file = r'c:\Users\Youcode\Desktop\Terravesion AI\data\raw\lst\MOD11A2.061_1km_aid0001.nc'

audit = {
    'NDVI': {},
    'LST': {}
}

# Audit NDVI
try:
    ds_ndvi = xr.open_dataset(ndvi_file)
    audit['NDVI']['file_name'] = os.path.basename(ndvi_file)
    audit['NDVI']['file_size_bytes'] = os.path.getsize(ndvi_file)
    audit['NDVI']['variables'] = list(ds_ndvi.data_vars.keys())
    audit['NDVI']['dimensions'] = dict(ds_ndvi.sizes)
    
    # Identify NDVI variable
    var_ndvi = None
    for v in ds_ndvi.data_vars:
        if 'NDVI' in v.upper():
            var_ndvi = v
            break
            
    if var_ndvi:
        audit['NDVI']['main_variable'] = var_ndvi
        da = ds_ndvi[var_ndvi]
        audit['NDVI']['data_type'] = str(da.dtype)
        audit['NDVI']['fill_value'] = da.attrs.get('_FillValue', None)
        audit['NDVI']['scale_factor'] = da.attrs.get('scale_factor', None)
        audit['NDVI']['offset'] = da.attrs.get('add_offset', None)
        audit['NDVI']['valid_min'] = float(da.min().values) if 'time' in da.dims else None
        audit['NDVI']['valid_max'] = float(da.max().values) if 'time' in da.dims else None
        
    if 'time' in ds_ndvi.dims:
        audit['NDVI']['temporal_observations'] = ds_ndvi.sizes['time']
        audit['NDVI']['date_range'] = [str(ds_ndvi['time'].values.min()), str(ds_ndvi['time'].values.max())]
    
    if 'lat' in ds_ndvi.dims and 'lon' in ds_ndvi.dims:
        audit['NDVI']['lat_range'] = [float(ds_ndvi['lat'].min()), float(ds_ndvi['lat'].max())]
        audit['NDVI']['lon_range'] = [float(ds_ndvi['lon'].min()), float(ds_ndvi['lon'].max())]
        
    audit['NDVI']['crs'] = ds_ndvi.attrs.get('crs', 'Not explicitly stated in global attrs')
    ds_ndvi.close()
except Exception as e:
    audit['NDVI']['error'] = str(e)

# Audit LST
try:
    ds_lst = xr.open_dataset(lst_file)
    audit['LST']['file_name'] = os.path.basename(lst_file)
    audit['LST']['file_size_bytes'] = os.path.getsize(lst_file)
    audit['LST']['variables'] = list(ds_lst.data_vars.keys())
    audit['LST']['dimensions'] = dict(ds_lst.sizes)
    
    # Identify LST variable
    var_lst = None
    for v in ds_lst.data_vars:
        if 'LST' in v.upper() and 'DAY' in v.upper():
            var_lst = v
            break
            
    if var_lst:
        audit['LST']['main_variable'] = var_lst
        da = ds_lst[var_lst]
        audit['LST']['data_type'] = str(da.dtype)
        audit['LST']['fill_value'] = da.attrs.get('_FillValue', None)
        audit['LST']['scale_factor'] = da.attrs.get('scale_factor', None)
        audit['LST']['offset'] = da.attrs.get('add_offset', None)
        audit['LST']['unit'] = da.attrs.get('units', 'Unknown')
        audit['LST']['valid_min'] = float(da.min().values) if 'time' in da.dims else None
        audit['LST']['valid_max'] = float(da.max().values) if 'time' in da.dims else None
        
    if 'time' in ds_lst.dims:
        audit['LST']['temporal_observations'] = ds_lst.sizes['time']
        audit['LST']['date_range'] = [str(ds_lst['time'].values.min()), str(ds_lst['time'].values.max())]
    
    if 'lat' in ds_lst.dims and 'lon' in ds_lst.dims:
        audit['LST']['lat_range'] = [float(ds_lst['lat'].min()), float(ds_lst['lat'].max())]
        audit['LST']['lon_range'] = [float(ds_lst['lon'].min()), float(ds_lst['lon'].max())]

    audit['LST']['crs'] = ds_lst.attrs.get('crs', 'Not explicitly stated in global attrs')
    ds_lst.close()
except Exception as e:
    audit['LST']['error'] = str(e)

# Save JSON
out_json = r'c:\Users\Youcode\Desktop\Terravesion AI\data\audit_ndvi_lst.json'
with open(out_json, 'w') as f:
    json.dump(audit, f, indent=4)
print(f"JSON audit saved to {out_json}")

# Save MD
out_md = r'c:\Users\Youcode\Desktop\Terravesion AI\data\audit_ndvi_lst.md'
with open(out_md, 'w') as f:
    f.write("# Data Sufficiency Audit: NDVI & LST\n\n")
    for dataset, info in audit.items():
        f.write(f"## {dataset}\n")
        for k, v in info.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n")
print(f"MD audit saved to {out_md}")
