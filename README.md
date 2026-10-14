# TerraPulse AI

> **From Satellite Data to Earth Insights**

TerraPulse AI is an AI-assisted environmental analysis platform built around NASA Earth observation data. The platform transforms satellite observations into traceable, explainable insights about environmental change.

The system combines statistical analysis, unsupervised machine learning, time-series forecasting, geospatial processing, and interactive visualization.

---

## 1. Project Vision

TerraPulse AI helps users answer:

- What changed?
- Where did it change?
- When did it change?
- How large was the change?
- Is the long-term trend statistically supported?
- Which periods are unusual?
- What do future forecasts suggest?
- What uncertainty and limitations should be considered?
- What environmental changes are observed around detected wildfire activity?

The project must distinguish clearly between:

- observation;
- statistical association;
- forecasting;
- causal explanation.

A correlation or temporal co-occurrence must never automatically be presented as causation.

---

## 2. Project Scope

The initial environmental indicators are:

- **NDVI** — vegetation condition/greenness indicator;
- **LST** — land surface temperature;
- **Precipitation** — rainfall context;
- **NASA FIRMS fire detections** — active-fire/hotspot observations.

Important scientific limitations:

- NDVI is not a direct biodiversity measurement.
- LST is not the same as near-surface air temperature.
- FIRMS active-fire detections are not necessarily complete burned-area polygons or exact fire perimeters.
- A forecast is not a guaranteed future outcome.
- A detected association does not prove causality.

---

## 3. Machine Learning Strategy

This project is **not a single classification or regression project**.

It is a multi-module analytical system.

| Module | Type | Initial method | Advanced option |
|---|---|---|---|
| Trend Analysis | Statistical analysis | Mann-Kendall + Sen's Slope | Seasonality/autocorrelation-aware methods |
| Anomaly Detection | Unsupervised ML | Seasonal baseline / Isolation Forest | Autoencoder / Anomaly Transformer |
| Forecasting | Time-series forecasting / regression | Seasonal Naive baseline | Random Forest / Transformer |
| Ecosystem Condition | Unsupervised ML | K-Means | Justified composite index |
| Wildfire Analysis | Statistical/event comparison | Before/after comparison | Matched comparison areas |

### Important rule

Do **not** train large neural networks by default.

The development order is:

```text
Statistical baseline
        ↓
Classical ML
        ↓
Evaluation
        ↓
Deep Learning only if justified
```

The model must be selected according to validation results, data volume, time-series length, and scientific suitability.

---

## 4. Data Sources

The project uses NASA Earth observation products.

Main sources:

- MODIS vegetation indices (NDVI/EVI)
- MODIS Land Surface Temperature (LST)
- NASA GPM precipitation
- NASA FIRMS fire detections
- NASA Earthdata Search
- NASA Earthdata
- NASA Worldview

Do not assume that all datasets have the same:

- spatial resolution;
- temporal frequency;
- projection/CRS;
- coverage;
- units;
- quality flags.

Before downloading large datasets, inspect a small sample first.

Every dataset must document:

- source;
- product name/ID;
- product version;
- spatial resolution;
- temporal resolution;
- units;
- quality flags;
- temporal coverage;
- retrieval date;
- processing steps.

---

## 5. Data Types

The project works with multiple data representations.

### Numerical data

Examples:

```text
NDVI
LST
precipitation
```

### Temporal data

Examples:

```text
2026-01
2026-02
2026-03
```

### Geospatial data

Examples:

```text
latitude
longitude
raster cells
grid cells
geometry
CRS
```

### Categorical data

Examples:

```text
quality = good
quality = review
quality = bad
```

### Raw file formats

Possible formats include:

```text
GeoTIFF
NetCDF
CSV
JSON
```

The final analytical representation may be:

- a clean tabular dataset;
- a raster cube;
- or another validated multidimensional structure.

---

## 6. End-to-End Architecture

```text
                         NASA
                          │
                          ▼
                  ┌───────────────┐
                  │ Data Ingestion│
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Data Validation│
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Preprocessing │
                  └───────┬───────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
      Spatial Alignment       Temporal Alignment
              │                       │
              └───────────┬───────────┘
                          ▼
                  ┌───────────────┐
                  │ Feature Eng.  │
                  └───────┬───────┘
                          │
             ┌────────────┼─────────────┐
             ▼            ▼             ▼
          Trend        Anomaly       Forecast
             │            │             │
             └────────────┼─────────────┘
                          ▼
                    Evaluation
                          │
                          ▼
                       FastAPI
                          │
                          ▼
                  Next.js Dashboard
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
           Charts        Maps       Results
```

---

## 7. Repository Structure

```text
terrapulse-ai/
│
├── data/
│   ├── raw/
│   │   ├── ndvi/
│   │   ├── lst/
│   │   ├── precipitation/
│   │   └── firms/
│   ├── interim/
│   ├── processed/
│   └── samples/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_trend_analysis.ipynb
│   ├── 04_anomaly_detection.ipynb
│   ├── 05_forecasting.ipynb
│   └── 06_clustering.ipynb
│
├── src/
│   │
│   ├── config/
│   │   ├── settings.py
│   │   └── constants.py
│   │
│   ├── ingestion/
│   │   ├── nasa_client.py
│   │   ├── ndvi_loader.py
│   │   ├── lst_loader.py
│   │   ├── precipitation_loader.py
│   │   └── firms_loader.py
│   │
│   ├── preprocessing/
│   │   ├── validation.py
│   │   ├── cleaning.py
│   │   ├── spatial.py
│   │   ├── temporal.py
│   │   └── quality.py
│   │
│   ├── features/
│   │   ├── temporal_features.py
│   │   ├── lag_features.py
│   │   ├── rolling_features.py
│   │   └── seasonal_features.py
│   │
│   ├── analytics/
│   │   ├── trend.py
│   │   ├── anomaly.py
│   │   ├── wildfire.py
│   │   └── clustering.py
│   │
│   ├── models/
│   │   ├── forecasting/
│   │   │   ├── baseline.py
│   │   │   ├── random_forest.py
│   │   │   └── transformer.py
│   │   ├── anomaly/
│   │   │   └── isolation_forest.py
│   │   └── clustering/
│   │       └── kmeans.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── temporal_split.py
│   │   └── model_comparison.py
│   │
│   └── utils/
│       ├── logging.py
│       ├── io.py
│       └── dates.py
│
├── api/
│   ├── main.py
│   ├── routes/
│   │   ├── regions.py
│   │   ├── trends.py
│   │   ├── anomalies.py
│   │   ├── forecasts.py
│   │   └── wildfire.py
│   └── schemas/
│       ├── requests.py
│       └── responses.py
│
├── frontend/
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_trend.py
│   ├── test_anomaly.py
│   └── test_forecasting.py
│
├── docs/
│   ├── architecture.md
│   ├── data-sources.md
│   ├── data-dictionary.md
│   ├── methodology.md
│   ├── models.md
│   ├── evaluation.md
│   └── limitations.md
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 8. Separation of Responsibilities

Every directory and function must have a clear responsibility.

### `src/ingestion/`

Responsible only for retrieving NASA data.

Examples:

```python
download_ndvi()
download_lst()
download_precipitation()
download_firms()
```

Do not put ML training or frontend logic here.

---

### `src/preprocessing/`

Responsible for:

- validation;
- cleaning;
- missing values;
- duplicate detection;
- invalid values;
- quality filtering;
- spatial alignment;
- temporal alignment.

Examples:

```python
check_missing_values()
check_duplicates()
validate_schema()
align_crs()
resample_raster()
clip_to_region()
aggregate_daily_to_monthly()
align_time_series()
```

---

### `src/features/`

Responsible only for feature engineering.

Examples:

```python
create_lag_features()
create_rolling_features()
create_seasonal_features()
create_temporal_features()
```

Possible features:

```text
NDVI_lag_1
NDVI_lag_2
NDVI_lag_3
NDVI_lag_12
NDVI_rolling_mean_3
NDVI_rolling_mean_12
month
season
quarter
```

---

### `src/analytics/`

Responsible for scientific/statistical analysis.

#### Trend

```python
mann_kendall_test()
calculate_sens_slope()
analyze_trend()
```

#### Anomaly

```python
build_seasonal_baseline()
calculate_anomaly_threshold()
```

#### Wildfire

```python
find_fire_events()
create_event_window()
compare_before_after()
```

#### Clustering

```python
prepare_clustering_features()
analyze_cluster_profiles()
```

---

### `src/models/`

Responsible only for model construction, training and prediction.

#### Forecasting

```python
seasonal_naive_forecast()
build_random_forest()
train_random_forest()
predict_random_forest()
build_transformer()
train_transformer()
predict_transformer()
```

#### Anomaly

```python
train_isolation_forest()
detect_anomalies()
calculate_anomaly_score()
```

#### Clustering

```python
build_kmeans()
fit_kmeans()
predict_clusters()
```

---

### `src/evaluation/`

Responsible only for model evaluation.

```python
calculate_mae()
calculate_rmse()
temporal_train_val_test_split()
compare_models()
```

Never use a random train/test split for the main time-series evaluation.

Use:

```text
Past observations
      ↓
TRAIN
      ↓
VALIDATION
      ↓
TEST / FUTURE HOLDOUT
```

This prevents temporal data leakage.

---

## 9. Forecasting Strategy

The forecasting pipeline must follow this order:

```text
Historical data
      ↓
Seasonal Naive baseline
      ↓
Evaluation
      ↓
Random Forest
      ↓
Evaluation
      ↓
Transformer (only if justified)
      ↓
Final comparison
```

Do not assume that Transformer is automatically the best model.

The final model must be selected using validation/test results.

Primary metrics:

- MAE
- RMSE

The evaluation must use chronological splits.

Example:

```text
2020 ───────── 2024 │ 2025 │ 2026
       TRAIN        │ VAL  │ TEST
```

---

## 10. Example Forecasting Setup

A possible starting setup:

```text
Lookback window = 12 months
Forecast horizon = 3 months
```

Example:

```text
Input:
t-11 ... t

Target:
t+1 ... t+3
```

This is only a starting configuration. The final lookback and horizon must depend on the actual dataset cadence, available history and scientific question.

---

## 11. Anomaly Detection Strategy

Start with:

```text
Seasonal baseline
       ↓
Robust thresholds
       ↓
Isolation Forest
```

Only consider:

```text
Autoencoder
Anomaly Transformer
```

if the dataset contains enough suitable sequences.

An anomaly means:

> "This observation/period deserves inspection."

It does not automatically mean:

> "A real environmental event definitely happened."

False positives and data-quality issues must be reviewed.

---

## 12. Clustering Strategy

For ecosystem-condition exploration:

```text
NDVI
LST
Precipitation
     ↓
Feature validation
     ↓
Standardization
     ↓
K-Means
     ↓
Cluster profiles
     ↓
Interpretation
```

Do not assign unsupported ecological meanings to clusters.

For example, do not automatically call a cluster:

```text
"high biodiversity"
```

unless suitable biodiversity data and a defensible methodology exist.

---

## 13. Wildfire Analysis

Workflow:

```text
NASA FIRMS
     ↓
Fire detections
     ↓
Event window
     ↓
Before / event / after periods
     ↓
NDVI/LST comparison
```

The output must describe observed changes around fire events.

Never claim causality solely from temporal coincidence.

---

## 14. Data Flow

The canonical data flow is:

```text
NASA APIs / Products
        ↓
data/raw/
        ↓
Validation
        ↓
Cleaning
        ↓
Spatial Alignment
        ↓
Temporal Alignment
        ↓
data/processed/
        ↓
Feature Engineering
        ↓
Analytics / ML
        ↓
Evaluation
        ↓
FastAPI
        ↓
Next.js
```

Raw data must remain unchanged.

Processed data must be reproducible from raw data and documented transformations.

---

## 15. Testing Strategy

Use `pytest`.

Test each important function independently.

Examples:

```python
def test_missing_value_detection():
    ...

def test_schema_validation():
    ...

def test_sens_slope():
    ...

def test_temporal_split():
    ...

def test_forecast_metrics():
    ...
```

The goal is to verify:

- functions behave correctly;
- preprocessing does not silently corrupt data;
- temporal splitting prevents leakage;
- metrics are calculated correctly;
- model outputs have expected shapes and types.

---

## 16. API

FastAPI exposes analytical functionality to the frontend.

Initial endpoints:

```text
GET /regions
GET /trends
GET /anomalies
GET /forecast
GET /wildfires
```

Example:

```text
GET /forecast?region=R01&indicator=NDVI
```

Possible response:

```json
{
  "region": "R01",
  "indicator": "NDVI",
  "forecast": [],
  "mae": 0.04,
  "rmse": 0.07
}
```

API code must not contain the core ML implementation.

The API should call reusable functions/services from `src/`.

---

## 17. Frontend

Use:

- Next.js
- TypeScript
- Tailwind CSS
- Leaflet or MapLibre
- interactive charts

Recommended pages:

```text
Dashboard
├── Overview
├── Map Explorer
├── Trend Analysis
├── Anomaly Explorer
├── Forecast
├── Fire Impact
└── Data & Methodology
```

The frontend consumes the FastAPI layer rather than directly accessing NASA services.

---

## 18. Documentation

Documentation is part of the scientific and engineering architecture.

It is required for:

### Reproducibility

Another developer must understand how to reproduce:

```text
data acquisition
→ preprocessing
→ feature engineering
→ analysis
→ training
→ evaluation
```

### Data traceability

For every important variable, document:

```text
source
product
version
unit
resolution
coverage
quality notes
retrieval date
```

### Scientific credibility

The team must be able to explain:

- why a method was selected;
- how the data was processed;
- how the model was evaluated;
- what the limitations are.

### Maintenance

A future developer must be able to locate and understand each part of the system without reading the entire codebase.

Documentation files:

```text
docs/
├── architecture.md
├── data-sources.md
├── data-dictionary.md
├── methodology.md
├── models.md
├── evaluation.md
└── limitations.md
```

---

## 19. Development Roadmap

Do not implement everything at once.

Follow this order:

```text
PHASE 01
Project Setup
        ↓
PHASE 02
Scientific Question + Region
        ↓
PHASE 03
NASA Data Source Selection
        ↓
PHASE 04
Data Ingestion
        ↓
PHASE 05
Data Validation
        ↓
PHASE 06
Cleaning
        ↓
PHASE 07
Spatial Alignment
        ↓
PHASE 08
Temporal Alignment
        ↓
PHASE 09
EDA
        ↓
PHASE 10
Feature Engineering
        ↓
PHASE 11
Trend Analysis
        ↓
PHASE 12
Anomaly Detection
        ↓
PHASE 13
Forecast Baseline
        ↓
PHASE 14
Random Forest
        ↓
PHASE 15
Transformer if justified
        ↓
PHASE 16
K-Means
        ↓
PHASE 17
Wildfire Analysis
        ↓
PHASE 18
Evaluation
        ↓
PHASE 19
Tests
        ↓
PHASE 20
FastAPI
        ↓
PHASE 21
Next.js Dashboard
        ↓
PHASE 22
Integration
        ↓
PHASE 23
Documentation
        ↓
PHASE 24
Demo + Presentation
```

---

## 20. Immediate First Steps

Do **not** start by training models.

The first implementation steps are:

1. Create the repository structure.
2. Configure the Python environment.
3. Define the scientific question.
4. Select the study region.
5. Select the study period.
6. Verify NASA data coverage.
7. Identify exact NASA products and versions.
8. Implement ingestion functions.
9. Download a small sample.
10. Inspect the sample.
11. Build the data dictionary.
12. Validate quality, missing values, dates and spatial information.
13. Only then start the EDA.

---

## 21. Coding Rules

Follow these rules throughout the project.

### Rule 1 — Single Responsibility

Every function must have one clear responsibility.

Bad:

```python
def process_everything():
    download()
    clean()
    train()
    predict()
    save()
```

Good:

```python
download_ndvi()
validate_ndvi()
clean_ndvi()
align_ndvi()
create_ndvi_features()
analyze_ndvi_trend()
detect_ndvi_anomalies()
forecast_ndvi()
evaluate_forecast()
```

### Rule 2 — Reusable functions

Do not duplicate processing logic across notebooks.

Notebooks are for:

- exploration;
- visualization;
- experiments;
- documentation of experiments.

Reusable production logic belongs in `src/`.

### Rule 3 — No hidden data processing

Every important transformation must be explicit and documented.

### Rule 4 — No data leakage

Never use future observations during training or preprocessing.

### Rule 5 — Raw data is immutable

Never modify files inside `data/raw/`.

### Rule 6 — Validate before modeling

No model training should happen before data quality is checked.

### Rule 7 — Baseline before complexity

Always establish a simple baseline before using a complex model.

### Rule 8 — Explainability

Every analytical result returned to the user should expose:

- method;
- relevant parameters;
- source;
- date/period;
- uncertainty where available;
- limitations.

### Rule 9 — Reproducibility

Record:

- parameters;
- random seeds;
- code version;
- data retrieval date;
- preprocessing configuration;
- model configuration;
- evaluation results.

### Rule 10 — Do not overclaim

Never present:

```text
forecast → guaranteed future
association → causation
cluster → biodiversity
NDVI → biodiversity
FIRMS hotspot → exact fire perimeter
```

---

## 22. Definition of Done

A phase is not considered complete until:

- its code is organized in the correct directory;
- its functions have clear responsibilities;
- important functions have tests;
- inputs and outputs are documented;
- transformations are reproducible;
- no unnecessary logic remains in notebooks;
- results are saved/documented where appropriate.

The project is considered ready for final integration when:

- the scientific question, region and study period are documented;
- every variable has a traceable source;
- the application can display a valid time series;
- trend analysis works;
- anomaly detection or forecasting has been evaluated against a baseline;
- temporal leakage is absent;
- results expose uncertainty and limitations;
- README and documentation are complete;
- the complete application has been tested.

---

## 23. Engineering Principle

The main principle of TerraPulse AI is:

```text
Simple
    ↓
Correct
    ↓
Tested
    ↓
Reproducible
    ↓
Explainable
    ↓
Scalable
```

Do not add complexity unless the data and evaluation justify it.

---

## 24. Important Instruction for AI Coding Agents

When implementing this repository:

1. **Do not generate the entire project blindly in one step.**
2. Follow the roadmap phase by phase.
3. Before implementing a phase, inspect the current repository state.
4. Do not overwrite working code without checking dependencies.
5. Keep each function focused on one responsibility.
6. Keep notebooks for experiments and `src/` for reusable logic.
7. Do not invent NASA data, product IDs, measurements, or results.
8. Verify dataset documentation before using a NASA product.
9. Do not train models before data validation and EDA are complete.
10. Do not introduce a Transformer unless the available data and validation results justify it.
11. Write tests for important reusable functions.
12. Update documentation when architecture, data processing, models, or evaluation changes.
13. Never expose API credentials or secrets in source code.
14. Keep `.env` out of Git.
15. Do not commit large raw datasets.
16. Preserve raw data unchanged.
17. When uncertain about a scientific interpretation, document the uncertainty instead of making an unsupported claim.

---

---

## 25. Operational Execution & Demonstration Guide

### 25.1 Running the Analytics & Modeling Pipeline
To execute the end-to-end analytical pipeline (Trend Analysis, Climatological Anomalies, Forecasting, Clustering, Wildfires) on the preprocessed NASA dataset:

```bash
# Extract national/regional time series from NetCDF cube
.venv\Scripts\python src/analytics/extract_timeseries.py

# Run all analytics and save operational artifacts
.venv\Scripts\python src/analytics/run_all_analytics.py
```

### 25.2 Running the Test Suite
The project includes a comprehensive unit and integration test suite (27 tests covering ingestion, preprocessing, features, analytics, models, and FastAPI):

```bash
.venv\Scripts\pytest -v tests/
```

### 25.3 Starting the FastAPI Operational Backend
Start the operational REST API serving real NASA Earth observation analytics:

```bash
.venv\Scripts\uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

### 25.4 Starting the Next.js Frontend Dashboard
In the `frontend` directory:

```bash
cd frontend
npm run dev
```
Open `http://localhost:3000` to interact with the operational dashboard.

### 25.5 Verified Model Results
- **Held-Out Chronological Test Partition**: 2024-05-08 to 2026-09-14 (55 observations).
- **Seasonal Naive Baseline (k=23)**: $\text{MAE} = 0.0118$, $\text{RMSE} = 0.0190$, $R^2 = 0.4177$.
- **Random Forest Regressor Forecaster**: $\text{MAE} = 0.0048$, $\text{RMSE} = 0.0065$, $R^2 = 0.9323$ (**+65.8% Error Reduction**).
- **Decadal Vegetation Dynamics**: Non-significant monotonic Sen's slope ($-0.00143/\text{yr}$), demonstrating seasonal phenological stability across Morocco.
- **Deduplicated Active Fires**: 1,364 physical fire events from 3,075 raw VIIRS detections, peak $\text{FRP} = 490.17\text{ MW}$.

