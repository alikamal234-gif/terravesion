import xarray as xr
import rioxarray
import geopandas as gpd
from rasterio.enums import Resampling
import numpy as np
import pandas as pd
import time

print("Testing 2-step pipeline...")
ndvi_file = r'data/raw/ndvi/MOD13Q1.061_250m_aid0001.nc'
lst_file = r'data/raw/lst/MOD11A2.061_1km_aid0001.nc'
boundary_file = r'data/raw/morocco_boundary.geojson'

# 1. Open datasets with mask_and_scale=False
ds_ndvi = xr.open_dataset(ndvi_file, mask_and_scale=False)
ds_lst = xr.open_dataset(lst_file, mask_and_scale=False)

crs_wkt = ds_ndvi['crs'].attrs.get('spatial_ref') or ds_ndvi['crs'].attrs.get('crs_wkt')
print("CRS verified.")

# 2. Boundary
gdf = gpd.read_file(boundary_file).to_crs(crs_wkt)
minx, miny, maxx, maxy = gdf.total_bounds

# 3. Crop LST template
x_mask = (ds_lst.xdim >= minx - 1000) & (ds_lst.xdim <= maxx + 1000)
y_mask = (ds_lst.ydim >= miny - 1000) & (ds_lst.ydim <= maxy + 1000)

target_template = ds_lst['LST_Day_1km'].isel(time=0).sel(xdim=x_mask, ydim=y_mask)
target_template = target_template.rio.set_spatial_dims(x_dim="xdim", y_dim="ydim").rio.write_crs(crs_wkt)

# Mask geometry on template
clipped_template = target_template.rio.clip(gdf.geometry, drop=False)
morocco_mask = ~np.isnan(clipped_template.values)
print(f"Template shape: {target_template.shape}, Valid Morocco pixels: {morocco_mask.sum()} / {morocco_mask.size} ({morocco_mask.sum()/morocco_mask.size*100:.1f}%)")

# 4. Test LST slice
raw_lst = ds_lst['LST_Day_1km'].isel(time=0).sel(xdim=x_mask, ydim=y_mask).values
raw_qc = ds_lst['QC_Day'].isel(time=0).sel(xdim=x_mask, ydim=y_mask).values
valid_lst = (raw_lst != 0) & (raw_lst != 65535) & ((raw_qc & 3) <= 1) & morocco_mask
lst_k = np.where(valid_lst, raw_lst * 0.02, np.nan).astype(np.float32)
print(f"LST test slice: min={np.nanmin(lst_k):.1f}K, max={np.nanmax(lst_k):.1f}K, mean={np.nanmean(lst_k):.1f}K")

# 5. Test NDVI slice
t0 = time.time()
raw_ndvi = ds_ndvi['_250m_16_days_NDVI'].isel(time=0)
valid_ndvi = xr.where((raw_ndvi >= -2000) & (raw_ndvi <= 10000), raw_ndvi.astype(np.float32) * 0.0001, np.nan)
valid_ndvi = valid_ndvi.rio.set_spatial_dims(x_dim="xdim", y_dim="ydim").rio.write_crs(crs_wkt)
resampled_ndvi = valid_ndvi.rio.reproject_match(target_template, resampling=Resampling.average)
ndvi_vals = resampled_ndvi.values
ndvi_vals[~morocco_mask] = np.nan
print(f"NDVI test slice ({time.time()-t0:.2f}s): min={np.nanmin(ndvi_vals):.3f}, max={np.nanmax(ndvi_vals):.3f}, mean={np.nanmean(ndvi_vals):.3f}")

print("Test passed successfully!")
