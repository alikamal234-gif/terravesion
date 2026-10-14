# TerraPulse AI — Architecture Documentation

## 1. System Overview
TerraPulse AI is an operational environmental monitoring and forecasting platform for the Kingdom of Morocco, powered by NASA Earth Observation data (MODIS Terra & VIIRS Active Fires). The system integrates a robust Python analytics engine, a high-performance FastAPI backend, and a modern Next.js dashboard.

```
                    ┌────────────────────────────┐
                    │      NASA LP DAAC / FIRMS   │
                    │   MOD13Q1, MOD11A2, VIIRS  │
                    └──────────────┬─────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────┐
                    │  Preprocessing & Alignment │
                    │   Masking, Resampling, QC  │
                    └──────────────┬─────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────┐
                    │ Analysis-Ready NetCDF Cube │
                    │ terra_pulse_morocco_clean  │
                    └──────┬──────────────┬──────┘
                           │              │
             ┌─────────────┴────┐    ┌────┴────────────────┐
             ▼                  ▼    ▼                     ▼
    ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
    │ Trend Analysis │ │ Anomaly Detect │ │ ML Forecasting │
    │ Seasonal MK    │ │ Climatological │ │ Random Forest  │
    └────────┬───────┘ └────────┬───────┘ └────────┬───────┘
             │                  │                  │
             └─────────────┬────┴──────────────────┘
                           ▼
                    ┌────────────────────────────┐
                    │       FastAPI Backend      │
                    │    api/ (REST Endpoints)   │
                    └──────────────┬─────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────┐
                    │      Next.js Frontend      │
                    │   frontend/ (Dashboard)    │
                    └────────────────────────────┘
```

## 2. Directory & Module Responsibilities

- **`data/raw/`**: Source NetCDF files and geojson boundary (immutable).
  - `data/raw/ndvi/MOD13Q1.061_250m_aid0001.nc`
  - `data/raw/lst/MOD11A2.061_1km_aid0001.nc`
  - `data/raw/morocco_boundary.geojson`
  - `data/raw/firms/`: Raw VIIRS active fire CSVs.
- **`data/interim/`**: Intermediate processed data.
  - `data/interim/firms_morocco_deduplicated.csv`: Spatiotemporally deduplicated FIRMS events (1 km, 2 h window).
- **`data/processed/`**: Analysis-ready raster and precomputed analytical artifacts.
  - `terra_pulse_morocco_clean.nc`: 1.01 GB compressed NetCDF cube (time: 271, ydim: 1807, xdim: 1802).
  - `morocco_timeseries.csv`: National spatial mean and variance across 11.75 years.
  - `trend_results.json`, `anomaly_summary.json`, `clustering_profiles.json`, `wildfire_summary.json`.
  - `checkpoints/rf_ndvi_forecaster.pkl`: Trained Random Forest model.
- **`src/ingestion/`**: Data discovery, loading, and deduplication (`firms_loader.py`).
- **`src/preprocessing/`**: QC bitmasking, physical range validation, spatial/temporal resampling.
- **`src/features/`**: Leak-free temporal cyclical, lag, and backward-looking rolling features (`shift=1`).
- **`src/analytics/`**: Seasonal Mann-Kendall, Climatological Anomaly, Clustering, Wildfire before/after.
- **`src/models/`**: Seasonal Naive Baseline ($k=23$), Random Forest Forecaster, Time-Series Transformer, K-Means.
- **`src/evaluation/`**: MAE, RMSE, R², Chronological splitting, and model benchmarking.
- **`api/`**: FastAPI operational routes (`/regions`, `/trends`, `/anomalies`, `/forecasts`, `/wildfire`, `/clustering`).
- **`frontend/`**: Next.js 16 + React 19 interactive operational dashboard.
- **`tests/`**: Pytest test suite covering ingestion, preprocessing, features, analytics, models, and API.
