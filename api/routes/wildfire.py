from fastapi import APIRouter, HTTPException, Query
import os
import json
import pandas as pd
import logging
from typing import Optional
from api.schemas.responses import WildfireResponse, WildfireEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/wildfire", tags=["Wildfire"])

@router.get("", response_model=WildfireResponse)
async def get_wildfire_events(
    min_frp: float = Query(0.0, ge=0.0, description="Minimum Fire Radiative Power (MW)"),
    start_date: Optional[str] = Query(None, description="Start date filter YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date filter YYYY-MM-DD"),
    limit: int = Query(100, ge=1, le=1500, description="Max events to return")
):
    """
    Returns real deduplicated NASA FIRMS fire detections across Morocco.
    """
    csv_file = "data/interim/firms_morocco_deduplicated.csv"
    summary_file = "data/processed/wildfire_summary.json"

    if not os.path.exists(csv_file):
        raise HTTPException(status_code=404, detail="Deduplicated FIRMS dataset not found.")

    df = pd.read_csv(csv_file)
    frp_col = "frp_total_mw" if "frp_total_mw" in df.columns else "frp"
    frp_max_col = "frp_max_mw" if "frp_max_mw" in df.columns else frp_col

    if min_frp > 0:
        df = df[df[frp_col] >= min_frp]
    if start_date:
        df = df[df["acq_date"] >= start_date]
    if end_date:
        df = df[df["acq_date"] <= end_date]

    df = df.sort_values(by=frp_col, ascending=False)
    total_matching = len(df)
    subset = df.head(limit)

    events = []
    for idx, row in subset.iterrows():
        events.append(
            WildfireEvent(
                event_id=str(row.get("event_id", f"E{idx}")),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                acq_date=str(row["acq_date"]),
                acq_time=str(row.get("acq_time", "")) if pd.notna(row.get("acq_time")) else None,
                satellite=str(row.get("satellites", row.get("satellite", "VIIRS"))),
                frp=round(float(row.get(frp_col, 0.0)), 2),
                frp_total_mw=round(float(row.get("frp_total_mw", row.get(frp_col, 0.0))), 2),
                frp_max_mw=round(float(row.get("frp_max_mw", 0.0)), 2),
                confidence=str(row.get("confidence", "nominal"))
            )
        )

    return WildfireResponse(
        total_events=total_matching,
        date_range={
            "start": str(df["acq_date"].min()) if not df.empty else "N/A",
            "end": str(df["acq_date"].max()) if not df.empty else "N/A",
        },
        mean_frp=round(float(df[frp_col].mean()), 2) if not df.empty else 0.0,
        max_frp=round(float(df[frp_max_col].max()), 2) if not df.empty else 0.0,
        peak_cluster_frp=round(float(df[frp_col].max()), 2) if not df.empty else 0.0,
        events=events
    )

