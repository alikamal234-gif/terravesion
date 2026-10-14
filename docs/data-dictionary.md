# TerraPulse AI — Data Dictionary & Variable Schema

This document details the scientific metadata, data types, physical units, scaling conventions, quality bitmasks, and storage schemas for all raw, intermediate, analysis-ready, and analytical datasets in TerraPulse AI.

---

## 1. Raw Earth Observation Products

### 1.1 Vegetation Indices (NDVI) — MOD13Q1 Version 061
- **Platform**: NASA Terra Satellite (AM overpass, ~10:30 local solar time)
- **Sensor**: Moderate Resolution Imaging Spectroradiometer (MODIS)
- **Product ID**: `MOD13Q1.061`
- **Cadence**: 16-day composite (derived from 16 daily 250m surface reflectance observations selecting optimal cloud-free, low-view-angle pixels)
- **Nominal Resolution**: 250 meters
- **Projection**: MODIS Sinusoidal Projection (SR-ORG:6974 / EPSG:9001)

| Variable Name | NetCDF Type | Physical Unit | Valid Range (Raw) | Valid Range (Physical) | Fill Value | Scale Factor | Add Offset | Description |
|---|---|---|---|---|---|---|---|---|
| `_250m_16_days_NDVI` | `int16` | Dimensionless index | `[-2000, 10000]` | `[-0.2000, 1.0000]` | `-3000` | `0.0001` | `0.0` | Normalized Difference Vegetation Index: $(NIR - Red) / (NIR + Red)$ |
| `_250m_16_days_EVI` | `int16` | Dimensionless index | `[-2000, 10000]` | `[-0.2000, 1.0000]` | `-3000` | `0.0001` | `0.0` | Enhanced Vegetation Index |
| `_250m_16_days_VI_Quality` | `uint16` | Bitmask | `[0, 65535]` | Bit flags | `65535` | `None` | `None` | 16-bit MODIS VI quality assurance bitmask |
| `time` | `int32` | Seconds | `[4.7e8, 8.4e8]` | `2014-12-19` to `2026-09-14` | None | None | None | Seconds since `2000-01-01 00:00:00 UTC` |
| `xdim` | `float64` | Meters | `[-1.93e6, 0.0]` | Sinusoidal easting | None | None | None | Projected East-West spatial coordinate |
| `ydim` | `float64` | Meters | `[2.22e6, 4.0e6]` | Sinusoidal northing | None | None | None | Projected North-South spatial coordinate |

#### MOD13Q1 VI_Quality Bitmask Breakdown:
- **Bits 0–1 (MODLAND QA)**:
  - `00`: VI produced with good quality.
  - `01`: VI produced, check other QA.
  - `10`: Produced, but cloud-contaminated.
  - `11`: Not produced due to target errors.
- **Bits 2–5 (VI Usefulness)**: `0000` = Highest quality, `1111` = Not produced.
- **Bits 6–7 (Aerosol Quantity)**: `00` = Climatology, `01` = Low, `10` = Average, `11` = High.
- **Bit 10 (Mixed Clouds)**: `0` = No, `1` = Yes.
- **Bit 15 (Possible shadow)**: `0` = No, `1` = Yes.

---

### 1.2 Land Surface Temperature (LST) — MOD11A2 Version 061
- **Platform**: NASA Terra Satellite (AM overpass)
- **Sensor**: MODIS
- **Product ID**: `MOD11A2.061`
- **Cadence**: 8-day composite (simple average of clear-sky daily MOD11A1 observations within the 8-day window)
- **Nominal Resolution**: 1,000 meters (~1 km)
- **Projection**: MODIS Sinusoidal Projection

| Variable Name | NetCDF Type | Physical Unit | Valid Range (Raw) | Valid Range (Physical) | Fill Value | Scale Factor | Add Offset | Description |
|---|---|---|---|---|---|---|---|---|
| `LST_Day_1km` | `uint16` / `int32` | Kelvin ($K$) | `[7500, 65535]` | `[150.0, 1310.7]` | `0` | `0.02` | `0.0` | Daytime Land Surface Temperature. Physical: $T_K = \text{raw} \times 0.02$. Celsius: $T_{°C} = T_K - 273.15$ |
| `QC_Day` | `uint8` | Bitmask | `[0, 255]` | Bit flags | None | None | None | 8-bit Daytime LST Quality Assurance flag |
| `Clear_sky_days` | `uint8` | Count | `[0, 8]` | `[0, 8]` days | `0` | None | None | Number of clear-sky days in composite |

#### MOD11A2 QC_Day Bitmask Breakdown:
- **Bits 0–1 (Mandatory QA Flag)**:
  - `00`: LST produced, good quality (no necessary calibration check).
  - `01`: LST produced, check other QA flags.
  - `10`: LST not produced due to cloud effects.
  - `11`: LST not produced due to other reasons.
- **Bits 2–3 (Data Quality Flag)**: `00` = Good quality, `01` = Other quality, `10` = Emissivity error, `11` = LST error.
- **Bits 4–5 (Emissivity Accuracy)**: `00` = Error $\le 0.01$, `01` = Error $\le 0.02$, `10` = Error $\le 0.04$, `11` = Error $> 0.04$.
- **Bits 6–7 (LST Error)**: `00` = Error $\le 1\text{ K}$, `01` = Error $\le 2\text{ K}$, `10` = Error $\le 3\text{ K}$, `11` = Error $> 3\text{ K}$.

---

### 1.3 Active Fire / Hotspots — NASA FIRMS VIIRS
- **Platforms**: Suomi-NPP, NOAA-20 (JPSS-1), NOAA-21 (JPSS-2)
- **Sensor**: Visible Infrared Imaging Radiometer Suite (VIIRS)
- **Product ID**: VIIRS 375m Active Fire (VNP14IMGTDL / VJ114IMGTDL)
- **Nominal Resolution**: 375 meters at nadir

| Field Name | Type | Physical Unit | Description |
|---|---|---|---|
| `latitude` | `float64` | Decimal Degrees | Latitude centroid of the 375m detection pixel (WGS84 EPSG:4326) |
| `longitude` | `float64` | Decimal Degrees | Longitude centroid of the 375m detection pixel (WGS84 EPSG:4326) |
| `bright_ti4` | `float64` | Kelvin ($K$) | VIIRS I-4 thermal channel (3.74 µm) brightness temperature |
| `scan` | `float64` | Kilometers | Pixel size along scan direction ($0.375\text{ km}$ nominal) |
| `track` | `float64` | Kilometers | Pixel size along track direction ($0.375\text{ km}$ nominal) |
| `acq_date` | `string` | ISO Date (`YYYY-MM-DD`) | Acquisition calendar date |
| `acq_time` | `string` | UTC Time (`HHMM`) | Acquisition time in UTC |
| `satellite` | `string` | Categorical | Satellite identifier (`N` = Suomi-NPP, `1` = NOAA-20, `2` = NOAA-21) |
| `confidence` | `string` | Categorical | Detection confidence quality tier (`l` = Low, `n` = Nominal, `h` = High) |
| `bright_ti5` | `float64` | Kelvin ($K$) | VIIRS I-5 thermal channel (11.45 µm) brightness temperature |
| `frp` | `float64` | Megawatts ($\text{MW}$) | Fire Radiative Power: instantaneous rate of radiant thermal energy emitted |
| `daynight` | `string` | Categorical | Solar illumination condition (`D` = Daytime, `N` = Nighttime) |

---

## 2. Processed & Analysis-Ready Datasets

### 2.1 Analysis-Ready NetCDF Cube (`data/processed/terra_pulse_morocco_clean.nc`)
- **Dimensions**:
  - `time`: 271 composite steps (16-day cadence, 2014-12-19 to 2026-09-14)
  - `ydim`: 1,807 grid rows (~1 km sinusoidal spacing)
  - `xdim`: 1,802 grid columns (~1 km sinusoidal spacing)
- **CRS**: MODIS Sinusoidal (`+proj=sinu +R=6371007.181 +nadgrids=@null +wktext`)
- **Spatial Extent**: Morocco national boundary polygon clipped via GeoBoundaries ADM0
- **Compression**: NetCDF-4 zlib compression level 4 with bit-shuffle filter

| Variable Name | Dtype | Dimensions | Units | Valid Range | No-Data Value | Description |
|---|---|---|---|---|---|---|
| `NDVI` | `float32` | `(time, ydim, xdim)` | Dimensionless | `[-0.20, 1.00]` | `NaN` | Area-weighted aggregated 1km NDVI with QA mask applied |
| `LST_Day_Celsius` | `float32` | `(time, ydim, xdim)` | Degrees Celsius (°C) | `[-15.0, 75.0]` | `NaN` | Coincident 16-day mean LST in °C, QC bitmasked |
| `morocco_mask` | `uint8` | `(ydim, xdim)` | Boolean flag | `{0, 1}` | `0` | Geographic mask (`1` = Inside Morocco land territory, `0` = Ocean / Outside) |

---

### 2.2 National Aggregated Time Series (`data/processed/morocco_timeseries.csv`)
- **Rows**: 271 observations
- **Temporal Span**: December 19, 2014 – September 14, 2026

| Column Name | Dtype | Unit | Min | Max | Mean | Description |
|---|---|---|---|---|---|---|
| `timestamp` | `datetime64[ns]` | ISO Date | 2014-12-19 | 2026-09-14 | — | 16-day composite start date |
| `ndvi_mean` | `float64` | Index | 0.1294 | 0.2589 | 0.1843 | Spatial mean NDVI across all valid terrestrial Morocco pixels |
| `ndvi_std` | `float64` | Index | 0.0512 | 0.1084 | 0.0765 | Spatial standard deviation of terrestrial NDVI |
| `lst_mean_celsius`| `float64` | °C | 14.26 | 45.53 | 31.05 | Spatial mean daytime LST across valid cloud-free pixels |
| `lst_std_celsius` | `float64` | °C | 4.12 | 9.87 | 6.74 | Spatial standard deviation of daytime LST |
| `valid_pixels_ndvi`| `int64`| Pixels | 812,045 | 1,024,880| 978,410 | Total valid pixels contributing to NDVI spatial mean |
| `valid_pixels_lst` | `int64` | Pixels | 450,110 | 995,320 | 810,240 | Total valid clear-sky pixels contributing to LST mean |

---

### 2.3 Spatiotemporally Deduplicated Wildfire Events (`data/interim/firms_morocco_deduplicated.csv`)
- **Total Physical Events**: 1,364 (deduplicated from 3,075 raw VIIRS detections)
- **Clustering Rule**: Spatial proximity $\le 1,000\text{ m}$ (Haversine distance) and temporal window $\le 2.0\text{ hours}$.

| Column Name | Dtype | Unit | Description |
|---|---|---|---|
| `event_id` | `string` | Unique ID | Synthesized unique identifier (`WF-0001`, `WF-0002`, etc.) |
| `cluster_id` | `int64` | Integer | DBSCAN spatiotemporal cluster index |
| `latitude` | `float64` | Degrees | Centroid latitude of cluster detections |
| `longitude` | `float64` | Degrees | Centroid longitude of cluster detections |
| `acq_date` | `string` | `YYYY-MM-DD` | Date of earliest detection in cluster |
| `frp_total_mw` | `float64` | Megawatts ($\text{MW}$) | Cumulative Fire Radiative Power summed across all cluster pixels |
| `frp_max_mw` | `float64` | Megawatts ($\text{MW}$) | Peak single-pixel Fire Radiative Power in cluster |
| `detection_count`| `int64`| Count | Number of raw 375m pixels constituting the unified physical event |
| `satellites` | `string` | Text list | Comma-delimited list of contributing satellites (e.g., `N, 1`) |
| `confidence` | `string` | Categorical | Highest confidence flag among cluster pixels (`h` or `n`) |

---

## 3. Feature Engineering Schema

Generated in `src/features/` with strict temporal causality:

| Feature Name | Origin / Formula | Range | Leak-Free Guarantee |
|---|---|---|---|
| `doy_sin` | $\sin(2\pi \cdot \text{DOY} / 365.25)$ | `[-1.0, 1.0]` | Derived strictly from calendar date $t$ |
| `doy_cos` | $\cos(2\pi \cdot \text{DOY} / 365.25)$ | `[-1.0, 1.0]` | Derived strictly from calendar date $t$ |
| `cycle_sin` | $\sin(2\pi \cdot k / 23)$ where $k \in [0, 22]$ | `[-1.0, 1.0]` | Derived from 23-period annual composite index |
| `cycle_cos` | $\cos(2\pi \cdot k / 23)$ | `[-1.0, 1.0]` | Derived from 23-period annual composite index |
| `ndvi_lag_1` | $y_{t-1}$ | `[0.12, 0.26]` | Shifted backward by 1 step (16 days prior) |
| `ndvi_lag_2` | $y_{t-2}$ | `[0.12, 0.26]` | Shifted backward by 2 steps (32 days prior) |
| `ndvi_lag_3` | $y_{t-3}$ | `[0.12, 0.26]` | Shifted backward by 3 steps (48 days prior) |
| `ndvi_lag_23` | $y_{t-23}$ | `[0.12, 0.26]` | Exactly 1 annual cycle prior (~368 days prior) |
| `ndvi_rolling_mean_3`| $\text{mean}(y_{t-1}, y_{t-2}, y_{t-3})$ | `[0.12, 0.26]` | Rolling window 3 with `shift=1` (no lookahead) |
| `ndvi_rolling_mean_6`| $\text{mean}(y_{t-1}, \dots, y_{t-6})$ | `[0.12, 0.26]` | Rolling window 6 with `shift=1` (no lookahead) |
| `lst_lag_1` | $\text{LST}_{t-1}$ | `[14.0, 46.0]` | Coincident LST shifted backward by 1 step |
