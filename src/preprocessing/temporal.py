import pandas as pd
import xarray as xr
import logging

logger = logging.getLogger(__name__)

class TemporalAligner:
    """
    Handles temporal alignment and resampling of datasets.
    """
    @staticmethod
    def set_datetime_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """
        Converts a column to datetime and sets it as the index.
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
        df.sort_index(inplace=True)
        logger.info(f"Set '{time_col}' as datetime index.")
        return df

    @staticmethod
    def aggregate_daily_to_monthly(df: pd.DataFrame, agg_func: str = "mean") -> pd.DataFrame:
        """
        Resamples a daily time-series to monthly frequency.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")
            
        resampled = df.resample('ME').agg(agg_func)
        logger.info(f"Resampled data to monthly frequency using '{agg_func}'.")
        return resampled

    @staticmethod
    def align_lst_to_ndvi_windows(da_lst: xr.DataArray, ndvi_times: pd.DatetimeIndex) -> xr.DataArray:
        """
        Aggregates LST observations into intervals defined by NDVI composite timestamps.
        For each NDVI observation T_i, computes the mean of valid LST observations in [T_i, T_{i+1}).
        If no LST observation is present, returns NaN.
        """
        import numpy as np
        
        # Ensure LST time coordinates are pandas DatetimeIndex for exact interval comparisons
        lst_times = pd.to_datetime(da_lst.time.values)
        
        aligned_slices = []
        for i, t_start in enumerate(ndvi_times):
            if i < len(ndvi_times) - 1:
                t_end = ndvi_times[i + 1]
            else:
                t_end = t_start + pd.Timedelta(days=16)
                
            mask = (lst_times >= t_start) & (lst_times < t_end)
            matching_indices = np.where(mask)[0]
            
            if len(matching_indices) == 0:
                # No LST observation available: slice with all NaNs
                empty_slice = xr.full_like(da_lst.isel(time=0), fill_value=np.nan).assign_coords(time=t_start)
                aligned_slices.append(empty_slice)
            elif len(matching_indices) == 1:
                single_slice = da_lst.isel(time=matching_indices[0]).assign_coords(time=t_start)
                aligned_slices.append(single_slice)
            else:
                window_mean = da_lst.isel(time=matching_indices).mean(dim='time', skipna=True).assign_coords(time=t_start)
                aligned_slices.append(window_mean)
                
        da_aligned = xr.concat(aligned_slices, dim='time')
        logger.info(f"Aligned LST to {len(ndvi_times)} NDVI periods.")
        return da_aligned
