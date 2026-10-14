from fastapi import APIRouter, HTTPException
import json
import os
import pandas as pd
import logging
from api.schemas.responses import RegionOverviewResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/regions", tags=["Regions"])

@router.get("/overview", response_model=RegionOverviewResponse)
async def get_region_overview():
    """
    Returns spatial and environmental metadata overview for the target region (Morocco).
    """
    ts_file = "data/processed/morocco_timeseries.csv"
    if not os.path.exists(ts_file):
        raise HTTPException(status_code=404, detail="Processed time-series data not found.")

    df = pd.read_csv(ts_file)
    if df.empty:
        raise HTTPException(status_code=500, detail="Processed time-series data is empty.")

    latest_row = df.iloc[-1]
    
    return RegionOverviewResponse(
        region_id="morocco",
        region_name="Kingdom of Morocco (National Domain)",
        bounding_box={
            "min_lon": -17.10,
            "min_lat": 20.76,
            "max_lon": -1.02,
            "max_lat": 35.93
        },
        crs="MODIS Sinusoidal (SR-ORG:6974 / EPSG:4326 equivalent)",
        temporal_coverage={
            "start_date": str(df["timestamp"].iloc[0]),
            "end_date": str(df["timestamp"].iloc[-1])
        },
        total_observations=len(df),
        latest_observation_date=str(latest_row["timestamp"]),
        latest_ndvi=float(latest_row["ndvi_mean"]),
        latest_lst_celsius=float(latest_row["lst_mean_celsius"]),
        summary_statistics={
            "ndvi_min": round(float(df["ndvi_mean"].min()), 4),
            "ndvi_max": round(float(df["ndvi_mean"].max()), 4),
            "ndvi_mean": round(float(df["ndvi_mean"].mean()), 4),
            "lst_min_celsius": round(float(df["lst_mean_celsius"].min()), 2),
            "lst_max_celsius": round(float(df["lst_mean_celsius"].max()), 2),
            "lst_mean_celsius": round(float(df["lst_mean_celsius"].mean()), 2),
        }
    )
