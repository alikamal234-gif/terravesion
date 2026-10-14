from fastapi import APIRouter, HTTPException, Query
import os
import json
import pandas as pd
import logging
from typing import Optional
from api.schemas.responses import TrendResponse, TimePoint

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/trends", tags=["Trends"])

@router.get("", response_model=TrendResponse)
async def get_trends(
    indicator: str = Query("NDVI", description="Target environmental indicator: 'NDVI' or 'LST'"),
    start_date: Optional[str] = Query(None, description="Optional filter start YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Optional filter end YYYY-MM-DD")
):
    """
    Returns empirical Mann-Kendall and Sen's Slope trend analysis for Morocco.
    """
    indicator = indicator.upper()
    if indicator not in ["NDVI", "LST"]:
        raise HTTPException(status_code=400, detail="Indicator must be either 'NDVI' or 'LST'.")

    trend_file = "data/processed/trend_results.json"
    ts_file = "data/processed/morocco_timeseries.csv"

    if not os.path.exists(trend_file) or not os.path.exists(ts_file):
        raise HTTPException(status_code=404, detail="Trend analysis results not yet computed.")

    with open(trend_file, "r") as f:
        trend_data = json.load(f)

    if indicator not in trend_data:
        raise HTTPException(status_code=404, detail=f"No trend results found for {indicator}.")

    item = trend_data[indicator]
    df = pd.read_csv(ts_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if start_date:
        df = df[df["timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["timestamp"] <= pd.to_datetime(end_date)]

    val_col = "ndvi_mean" if indicator == "NDVI" else "lst_mean_celsius"
    series_points = [
        TimePoint(timestamp=str(row["timestamp"])[:10], value=round(float(row[val_col]), 4))
        for _, row in df.iterrows()
    ]

    return TrendResponse(
        indicator=indicator,
        trend_direction=item["trend_direction"],
        sen_slope=item["sen_slope"],
        sen_slope_per_decade=round(item["sen_slope_per_year"] * 10, 4),
        p_value=item["p_value"],
        is_significant=item["is_significant"],
        tau=item["tau"],
        n_observations=item["n_observations"],
        interpretation=item["interpretation"],
        time_series=series_points
    )
