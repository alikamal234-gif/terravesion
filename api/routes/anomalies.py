from fastapi import APIRouter, HTTPException, Query
import os
import json
import pandas as pd
import logging
from api.schemas.responses import AnomalySummaryResponse, AnomalyPoint

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/anomalies", tags=["Anomalies"])

_if_cache = {}

@router.get("", response_model=AnomalySummaryResponse)
async def get_anomalies(
    indicator: str = Query("NDVI", description="Target environmental indicator: 'NDVI' or 'LST'"),
    threshold_sigma: float = Query(2.0, ge=1.0, le=4.0, description="Standard deviation anomaly threshold"),
    method: str = Query("climatological", description="Detection method: 'climatological' or 'isolation_forest'")
):
    """
    Returns climatological baseline anomaly detection or Isolation Forest ML anomaly alerts.
    """
    indicator = indicator.upper()
    if indicator not in ["NDVI", "LST"]:
        raise HTTPException(status_code=400, detail="Indicator must be either 'NDVI' or 'LST'.")

    summary_file = "data/processed/anomaly_summary.json"
    results_file = "data/processed/anomaly_results.csv"

    if not os.path.exists(summary_file) or not os.path.exists(results_file):
        raise HTTPException(status_code=404, detail="Anomaly results not yet computed.")

    df = pd.read_csv(results_file)
    prefix = "ndvi" if indicator == "NDVI" else "lst"
    total_obs = len(df)

    if method.lower() == "isolation_forest":
        if prefix not in _if_cache:
            from src.models.anomaly.isolation_forest import IsolationForestAnomalyDetector
            features = pd.DataFrame({
                "actual": df[f"{prefix}_actual"],
                "baseline": df[f"{prefix}_baseline"]
            })
            detector = IsolationForestAnomalyDetector(contamination=0.05, random_state=42)
            detector.train_isolation_forest(features)
            _if_cache[prefix] = detector.detect_anomalies(features)

        if_res = _if_cache[prefix]
        is_anomaly = if_res["is_anomaly"]
        total_anomalies = int(is_anomaly.sum())

        anomaly_points = []
        for idx, row in df.iterrows():
            is_anom = bool(if_res.loc[idx, "is_anomaly"])
            score = float(if_res.loc[idx, "anomaly_score"])
            severity = "severe" if is_anom else "normal"
            anomaly_points.append(
                AnomalyPoint(
                    timestamp=str(row["timestamp"])[:10],
                    actual_value=float(row[f"{prefix}_actual"]),
                    climatological_mean=float(row[f"{prefix}_baseline"]),
                    anomaly_delta=float(row[f"{prefix}_anomaly_delta"]),
                    z_score=round(score, 4),
                    is_anomaly=is_anom,
                    severity=severity
                )
            )

        return AnomalySummaryResponse(
            indicator=indicator,
            total_observations=total_obs,
            total_anomalies=total_anomalies,
            anomaly_rate_percent=round(total_anomalies / total_obs * 100, 2) if total_obs > 0 else 0.0,
            severe_negative_count=int((df[f"{prefix}_z_score"] <= -threshold_sigma).sum()),
            severe_positive_count=int((df[f"{prefix}_z_score"] >= threshold_sigma).sum()),
            methodology="Unsupervised Isolation Forest ML (contamination=0.05, n_estimators=100)",
            anomalies=anomaly_points
        )

    # Default: Climatological baseline filter based on requested threshold_sigma
    if threshold_sigma == 2.0 and f"{prefix}_is_anomaly" in df.columns:
        is_anomaly = df[f"{prefix}_is_anomaly"].astype(bool)
    else:
        is_anomaly = df[f"{prefix}_z_score"].abs() >= threshold_sigma
    total_anomalies = int(is_anomaly.sum())

    anomaly_points = []
    for _, row in df.iterrows():
        z = float(row[f"{prefix}_z_score"])
        is_anom = bool(row[f"{prefix}_is_anomaly"]) if (threshold_sigma == 2.0 and f"{prefix}_is_anomaly" in row) else (abs(z) >= threshold_sigma)
        
        severity = "normal"
        if is_anom:
            if z <= -3.0 or z >= 3.0:
                severity = "extreme"
            elif z <= -2.0 or z >= 2.0:
                severity = "severe"
            else:
                severity = "moderate"

        anomaly_points.append(
            AnomalyPoint(
                timestamp=str(row["timestamp"])[:10],
                actual_value=float(row[f"{prefix}_actual"]),
                climatological_mean=float(row[f"{prefix}_baseline"]),
                anomaly_delta=float(row[f"{prefix}_anomaly_delta"]),
                z_score=z,
                is_anomaly=is_anom,
                severity=severity
            )
        )

    return AnomalySummaryResponse(
        indicator=indicator,
        total_observations=total_obs,
        total_anomalies=total_anomalies,
        anomaly_rate_percent=round(total_anomalies / total_obs * 100, 2) if total_obs > 0 else 0.0,
        severe_negative_count=int((df[f"{prefix}_z_score"] <= -threshold_sigma).sum()),
        severe_positive_count=int((df[f"{prefix}_z_score"] >= threshold_sigma).sum()),
        methodology="Multi-year climatological baseline (k=23 annual cycle, strictly leak-free)",
        anomalies=anomaly_points
    )
