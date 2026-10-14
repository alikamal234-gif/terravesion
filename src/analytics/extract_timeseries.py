"""
Extracts national and regional time-series aggregates from the preprocessed NetCDF cube.
Saves results into data/processed/ for fast API querying, analytics, and modeling.
"""
import xarray as xr
import pandas as pd
import numpy as np
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def extract_national_timeseries(nc_path: str = "data/processed/terra_pulse_morocco_clean.nc", output_csv: str = "data/processed/morocco_timeseries.csv"):
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"Processed NetCDF file not found at {nc_path}")

    logger.info(f"Opening dataset: {nc_path}")
    # Open with chunking across time
    ds = xr.open_dataset(nc_path, chunks={"time": 10, "ydim": -1, "xdim": -1})
    
    times = ds.time.values
    logger.info(f"Computing spatial aggregates across {len(times)} time steps...")
    
    start_t = time.time()
    
    # Efficient spatial mean across ydim and xdim
    ndvi_mean = ds["NDVI"].mean(dim=["ydim", "xdim"], skipna=True).compute()
    lst_mean = ds["LST"].mean(dim=["ydim", "xdim"], skipna=True).compute()
    
    # Also standard deviation to gauge spatial variability
    ndvi_std = ds["NDVI"].std(dim=["ydim", "xdim"], skipna=True).compute()
    lst_std = ds["LST"].std(dim=["ydim", "xdim"], skipna=True).compute()
    
    elapsed = time.time() - start_t
    logger.info(f"Completed spatial aggregation in {elapsed:.2f} seconds.")
    
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([str(t)[:10] for t in times]),
        "ndvi_mean": ndvi_mean.values,
        "ndvi_std": ndvi_std.values,
        "lst_mean_celsius": lst_mean.values,
        "lst_std_celsius": lst_std.values,
    })
    
    # Round to reasonable precision
    df["ndvi_mean"] = df["ndvi_mean"].round(4)
    df["ndvi_std"] = df["ndvi_std"].round(4)
    df["lst_mean_celsius"] = df["lst_mean_celsius"].round(2)
    df["lst_std_celsius"] = df["lst_std_celsius"].round(2)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved national time series ({len(df)} rows) to {output_csv}")
    print(df.head(10))
    print(df.tail(10))
    return df

if __name__ == "__main__":
    extract_national_timeseries()
