import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

logger = logging.getLogger(__name__)

class WildfireAnalyzer:
    """
    Analytics logic for evaluating the impact of wildfires (NASA FIRMS)
    on environmental indicators (NDVI, LST) over time.
    """
    def __init__(self, crs_wkt: Optional[str] = None):
        # Default MODIS Sinusoidal projection parameters
        if crs_wkt is None:
            self.transformer = Transformer.from_crs("EPSG:4326", "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext", always_xy=True)
        else:
            self.transformer = Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)

    def latlon_to_sinusoidal(self, lat: float, lon: float) -> Tuple[float, float]:
        x, y = self.transformer.transform(lon, lat)
        return float(x), float(y)

    @staticmethod
    def find_fire_events(firms_df: pd.DataFrame, confidence_threshold: str = "n") -> pd.DataFrame:
        """
        Filters the FIRMS dataset to retain detections meeting or exceeding the confidence threshold.
        Confidence levels: 'l' (low) < 'n' (nominal) < 'h' (high).
        """
        conf_map = {"l": 1, "n": 2, "h": 3}
        target_score = conf_map.get(confidence_threshold, 2)
        
        if "confidence" in firms_df.columns:
            mask = firms_df["confidence"].map(conf_map).fillna(1) >= target_score
            filtered = firms_df[mask].copy()
            logger.info(f"Filtered {len(filtered)} fire events with confidence >= '{confidence_threshold}'.")
            return filtered
        return firms_df

    @staticmethod
    def create_event_window(event_date: pd.Timestamp, before_days: int = 48, after_days: int = 48) -> Dict[str, Any]:
        """
        Creates a temporal window around a specific fire event.
        Default 48 days corresponds to 3 full 16-day composite cycles before and after.
        """
        before_period = (event_date - pd.Timedelta(days=before_days), event_date - pd.Timedelta(days=1))
        after_period = (event_date + pd.Timedelta(days=1), event_date + pd.Timedelta(days=after_days))
        
        return {
            "event_date": event_date,
            "before_period": before_period,
            "after_period": after_period
        }

    @staticmethod
    def compare_before_after(indicator_series: pd.Series, event_window: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compares an environmental indicator (e.g., NDVI, LST) before and after the fire event.
        """
        if not isinstance(indicator_series.index, pd.DatetimeIndex):
            raise ValueError("Indicator series must have a DatetimeIndex.")
            
        before_start, before_end = event_window["before_period"]
        after_start, after_end = event_window["after_period"]
        
        before_data = indicator_series.loc[before_start:before_end].dropna()
        after_data = indicator_series.loc[after_start:after_end].dropna()
        
        before_mean = float(before_data.mean()) if not before_data.empty else float("nan")
        after_mean = float(after_data.mean()) if not after_data.empty else float("nan")
        
        if not np.isnan(before_mean) and not np.isnan(after_mean):
            abs_change = after_mean - before_mean
            pct_change = (abs_change / before_mean) * 100.0 if before_mean != 0 else float("nan")
        else:
            abs_change = float("nan")
            pct_change = float("nan")
            
        return {
            "before_mean": round(before_mean, 4),
            "after_mean": round(after_mean, 4),
            "absolute_change": round(abs_change, 4),
            "percent_change": round(pct_change, 2),
            "n_before_obs": len(before_data),
            "n_after_obs": len(after_data)
        }

    def analyze_event_impact(
        self,
        event: pd.Series,
        ds: xr.Dataset,
        before_days: int = 48,
        after_days: int = 48
    ) -> Dict[str, Any]:
        """
        Extracts coincident NDVI and LST pixel trajectories around a FIRMS event
        and measures vegetative and thermal impact.
        """
        lat = float(event["latitude"])
        lon = float(event["longitude"])
        t_event = pd.to_datetime(event.get("datetime_first_utc") or event["acq_date"])
        
        # Transform lat/lon to raster coordinates
        x_proj, y_proj = self.latlon_to_sinusoidal(lat, lon)
        
        # Sample nearest pixel
        pixel_data = ds.sel(xdim=x_proj, ydim=y_proj, method="nearest")
        t_index = pd.to_datetime(ds.time.values)
        
        ndvi_s = pd.Series(pixel_data["NDVI"].values, index=t_index)
        lst_s = pd.Series(pixel_data["LST"].values, index=t_index)
        
        window = self.create_event_window(t_event, before_days, after_days)
        ndvi_impact = self.compare_before_after(ndvi_s, window)
        lst_impact = self.compare_before_after(lst_s, window)
        
        return {
            "event_id": event.get("event_id", 0),
            "latitude": lat,
            "longitude": lon,
            "event_date": t_event.strftime("%Y-%m-%d"),
            "satellites": event.get("satellites", "VIIRS"),
            "frp_total_mw": float(event.get("frp_total_mw", 0)),
            "ndvi_impact": ndvi_impact,
            "lst_impact": lst_impact
        }
