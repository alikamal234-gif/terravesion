from fastapi import APIRouter, HTTPException, Query
import os
import json
import pickle
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import logging

from api.schemas.requests import ForecastRequest
from api.schemas.responses import (
    ForecastResponse,
    ForecastPoint,
    TimePoint,
    ModelMetrics,
    FeatureImportance,
)
from src.models.forecasting.baseline import SeasonalNaiveForecast

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/forecasts", tags=["Forecasts"])

FEATURE_DESCRIPTIONS = {
    "ndvi_mean_lag_1": "Lag-1 Autoregressive persistence (previous 16-day composite)",
    "lst_lag1": "Land Surface Temperature lag-1 (thermal coupling with vegetation)",
    "ndvi_mean_lag_23": "Annual seasonal cycle lag (same 16-day composite last year)",
    "cos_doy": "Harmonic day-of-year seasonal cycle component (cosine)",
    "ndvi_mean_rolling_mean_6": "96-day rolling average vegetative baseline",
    "cos_period_16d": "Harmonic 16-day composite period component (cosine)",
    "ndvi_mean_lag_2": "Lag-2 Autoregressive persistence (32 days ago)",
    "ndvi_mean_lag_3": "Lag-3 Autoregressive persistence (48 days ago)",
    "ndvi_mean_rolling_mean_3": "48-day rolling average vegetative baseline",
    "sin_period_16d": "Harmonic 16-day periodic cycle component (sine)",
    "period_16d": "Index of 16-day composite in annual cycle (0-22)",
    "sin_doy": "Harmonic day-of-year seasonal component (sine)",
}

def execute_forecast_pipeline(
    region: str = "morocco",
    indicator: str = "NDVI",
    model_name: str = "random_forest",
    steps: int = 23,
) -> ForecastResponse:
    """
    Executes actual backend model forecasting pipeline with empirical verification.
    """
    clean_indicator = indicator.strip().upper()
    if clean_indicator != "NDVI":
        raise HTTPException(
            status_code=400,
            detail=f"Forecasting for indicator '{indicator}' is not supported. Only 'NDVI' models are trained and validated."
        )

    clean_region = region.strip().lower()
    if clean_region != "morocco":
        raise HTTPException(
            status_code=400,
            detail=f"Geographic scope '{region}' is not supported. Only Morocco national domain ('morocco') is currently supported."
        )

    clean_model = model_name.strip().lower().replace("-", "_").replace(" ", "_")
    if clean_model not in ["random_forest", "seasonal_naive", "random_forest_regressor", "seasonal_naive_baseline"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is not supported. Supported models: 'random_forest' and 'seasonal_naive'."
        )

    if steps < 1 or steps > 46:
        raise HTTPException(
            status_code=400,
            detail="Prediction horizon steps must be between 1 and 46 (approx. 16 to 736 days)."
        )

    eval_file = "data/processed/evaluation/model_comparison.json"
    ts_file = "data/processed/morocco_timeseries.csv"
    rf_ckpt = "data/processed/checkpoints/rf_ndvi_forecaster.pkl"

    if not os.path.exists(eval_file) or not os.path.exists(ts_file):
        raise HTTPException(status_code=404, detail="Forecasting model data and evaluation artifacts not found.")

    with open(eval_file, "r") as f:
        eval_data = json.load(f)

    rf_m = eval_data.get("random_forest", {})
    base_m = eval_data.get("baseline_seasonal_naive", {})

    # Load time series
    ts_df = pd.read_csv(ts_file)
    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
    last_date = ts_df["timestamp"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=16), periods=steps, freq="16D")

    # Load recent historical points (last 46 points = ~2 years of history for context)
    hist_subset = ts_df.tail(46)
    historical_points = [
        TimePoint(timestamp=str(row["timestamp"])[:10], value=round(float(row["ndvi_mean"]), 4))
        for _, row in hist_subset.iterrows()
    ]

    forecast_points: List[ForecastPoint] = []
    feature_importances_list: Optional[List[FeatureImportance]] = None
    limitations: List[str] = []

    if clean_model in ["random_forest", "random_forest_regressor"]:
        if not os.path.exists(rf_ckpt):
            raise HTTPException(status_code=404, detail="Random Forest model checkpoint not found.")

        with open(rf_ckpt, "rb") as f:
            rf_model = pickle.load(f)

        history_ndvi = list(ts_df["ndvi_mean"].values)
        history_lst = list(ts_df["lst_mean_celsius"].values)
        rf_rmse = float(rf_m.get("rmse", 0.0065))

        for i, dt in enumerate(future_dates):
            doy = dt.dayofyear
            sin_doy = float(np.sin(2 * np.pi * doy / 365.25))
            cos_doy = float(np.cos(2 * np.pi * doy / 365.25))
            period_16d = min(int(doy // 16), 22)
            sin_period_16d = float(np.sin(2 * np.pi * period_16d / 23.0))
            cos_period_16d = float(np.cos(2 * np.pi * period_16d / 23.0))

            lag1 = float(history_ndvi[-1])
            lag2 = float(history_ndvi[-2])
            lag3 = float(history_ndvi[-3])
            lag23 = float(history_ndvi[-23])

            roll3 = float(np.mean(history_ndvi[-3:]))
            roll6 = float(np.mean(history_ndvi[-6:]))
            lst_lag1 = float(history_lst[-1])

            row = pd.DataFrame([{
                "sin_doy": sin_doy,
                "cos_doy": cos_doy,
                "period_16d": period_16d,
                "sin_period_16d": sin_period_16d,
                "cos_period_16d": cos_period_16d,
                "ndvi_mean_lag_1": lag1,
                "ndvi_mean_lag_2": lag2,
                "ndvi_mean_lag_3": lag3,
                "ndvi_mean_lag_23": lag23,
                "ndvi_mean_rolling_mean_3": roll3,
                "ndvi_mean_rolling_mean_6": roll6,
                "lst_lag1": lst_lag1
            }])

            pred_val = float(rf_model.predict(row)[0])
            pred_val = round(max(0.05, min(0.35, pred_val)), 4)
            lower_bound = round(max(0.0, pred_val - 1.96 * rf_rmse), 4)
            upper_bound = round(min(0.5, pred_val + 1.96 * rf_rmse), 4)

            forecast_points.append(
                ForecastPoint(
                    timestamp=dt.strftime("%Y-%m-%d"),
                    predicted_value=pred_val,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            )
            history_ndvi.append(pred_val)
            history_lst.append(history_lst[-23])

        # Extract authentic feature importances
        if hasattr(rf_model, "feature_importances_") and hasattr(rf_model, "feature_names_in_"):
            raw_imp = list(zip(rf_model.feature_names_in_, rf_model.feature_importances_))
            raw_imp.sort(key=lambda x: x[1], reverse=True)
            feature_importances_list = [
                FeatureImportance(
                    feature_name=fname,
                    importance=round(float(val), 4),
                    description=FEATURE_DESCRIPTIONS.get(fname, "Engineered predictor feature")
                )
                for fname, val in raw_imp
            ]

        model_label = "Random_Forest_Regressor"
        limitations = [
            "Recursive multi-step autoregressive forecasting accumulates variance across longer horizons (>12 steps).",
            "Trained on national aggregate NDVI (2014–2024); sub-regional microclimates may exhibit localized divergence.",
            "Assumes stationary lag relationships; cannot predict unprecedented external climate shocks or novel extreme weather events."
        ]

    else:
        # Seasonal Naive Baseline
        baseline = SeasonalNaiveForecast(seasonality=23)
        baseline.fit(ts_df.set_index("timestamp")["ndvi_mean"])
        naive_preds = baseline.predict(steps=steps, future_index=future_dates)
        naive_rmse = float(base_m.get("rmse", 0.0190))

        for dt, val in zip(future_dates, naive_preds.values):
            pred_val = round(float(val), 4)
            lower_bound = round(max(0.0, pred_val - 1.96 * naive_rmse), 4)
            upper_bound = round(min(0.5, pred_val + 1.96 * naive_rmse), 4)
            forecast_points.append(
                ForecastPoint(
                    timestamp=dt.strftime("%Y-%m-%d"),
                    predicted_value=pred_val,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            )

        model_label = "Seasonal_Naive_Baseline"
        limitations = [
            "Assumes exact repetition of the preceding annual cycle; cannot predict multi-year persistent droughts or decadal trends.",
            "Fixed 23-step periodicity (16-day composites); does not adapt to shifts in seasonal onset timing or prolonged dry spells.",
            "Non-parametric heuristic baseline; feature importance and parametric uncertainty bounds do not apply."
        ]

    comparison_summary = (
        f"Random Forest achieved RMSE of {rf_m.get('rmse', 0):.4f} (R2={rf_m.get('r2', 0):.4f}) "
        f"on held-out chronological test set ({eval_data['temporal_test_split']['test_start']} to {eval_data['temporal_test_split']['test_end']}). "
        f"Baseline Seasonal Naive achieved RMSE of {base_m.get('rmse', 0):.4f} (R2={base_m.get('r2', 0):.4f})."
    )

    training_period = {
        "train_start": "2014-01-01",
        "train_end": eval_data["temporal_test_split"]["test_start"],
        "n_train_composites": eval_data["temporal_test_split"]["n_train"],
        "test_start": eval_data["temporal_test_split"]["test_start"],
        "test_end": eval_data["temporal_test_split"]["test_end"],
        "n_test_composites": eval_data["temporal_test_split"]["n_test"],
    }

    return ForecastResponse(
        region=clean_region,
        indicator=clean_indicator,
        model_type=model_label,
        forecast_steps=len(forecast_points),
        forecast=forecast_points,
        historical=historical_points,
        model_metrics=ModelMetrics(
            mae=round(float(rf_m.get("mae", 0.0)), 4),
            rmse=round(float(rf_m.get("rmse", 0.0)), 4),
            r2=round(float(rf_m.get("r2", 0.0)), 4),
        ),
        baseline_metrics=ModelMetrics(
            mae=round(float(base_m.get("mae", 0.0)), 4),
            rmse=round(float(base_m.get("rmse", 0.0)), 4),
            r2=round(float(base_m.get("r2", 0.0)), 4),
        ),
        comparison_summary=comparison_summary,
        feature_importances=feature_importances_list,
        generation_timestamp=datetime.now(timezone.utc).isoformat(),
        training_period=training_period,
        limitations=limitations,
    )

@router.get("", response_model=ForecastResponse)
async def get_forecast(
    region: str = Query("morocco", description="Target region"),
    indicator: str = Query("NDVI", description="Target environmental indicator: 'NDVI'"),
    steps: int = Query(23, ge=1, le=46, description="Forecast steps ahead (23 = 1 year)"),
    model: str = Query("random_forest", description="Model: 'random_forest' or 'seasonal_naive'"),
):
    """
    Returns verified model forecasts and empirical evaluation metrics.
    """
    return execute_forecast_pipeline(region=region, indicator=indicator, model_name=model, steps=steps)

@router.post("/predict", response_model=ForecastResponse)
async def run_prediction(request: ForecastRequest):
    """
    Configurable endpoint to execute predictions with selected model and horizon parameters.
    """
    return execute_forecast_pipeline(
        region=request.region_id,
        indicator=request.indicator,
        model_name=request.model,
        steps=request.steps,
    )
