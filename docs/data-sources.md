# TerraPulse AI — Data Sources & Provenance

## 1. Primary Earth Observation Datasets

### 1.1 Vegetation Dynamics (NDVI) — MOD13Q1.061
- **Platform**: NASA Terra satellite
- **Instrument**: Moderate Resolution Imaging Spectroradiometer (MODIS)
- **Product**: MOD13Q1 Version 061 (Vegetation Indices 16-Day L3 Global 250m)
- **Variable**: `_250m_16_days_NDVI`
- **Data Type**: `int16`
- **Scale Factor**: `0.0001`
- **Valid Range**: `[-2000, 10000]` raw (`[-0.2, 1.0]` physical NDVI)
- **Fill Value**: `-3000`
- **Quality Band**: `_250m_16_days_VI_Quality` (16-bit assessment bitmask)
- **Temporal Coverage**: 2014-12-19 to 2026-09-14 (271 observations)
- **Spatial Resolution**: 250 m nominal, aggregated to 1 km via area-weighted averaging.

### 1.2 Thermal Land Surface Temperature (LST) — MOD11A2.061
- **Platform**: NASA Terra satellite
- **Instrument**: MODIS
- **Product**: MOD11A2 Version 061 (Land Surface Temperature and Emissivity 8-Day L3 Global 1km)
- **Variable**: `LST_Day_1km`
- **Data Type**: `int32`
- **Scale Factor**: `0.02`
- **Units**: Kelvin (converted to Celsius: $T_{°C} = T_K - 273.15$)
- **Valid Range**: `7500` to `65535` raw
- **Fill Value**: `0`
- **Sentinel Error Flag**: `65535` (uncalibrated / cloud obscured)
- **Quality Band**: `QC_Day` (8-bit assessment bitmask)
- **Alignment**: Aligned to coincident 16-day NDVI periods.

### 1.3 Active Fire Hotspots — NASA FIRMS VIIRS
- **Platform**: Suomi-NPP, NOAA-20 (JPSS-1), NOAA-21 (JPSS-2)
- **Instrument**: Visible Infrared Imaging Radiometer Suite (VIIRS)
- **Product**: Active Fire / Hotspot detections (375 m resolution)
- **Raw Detections Ingested**: 3,075 observations across Morocco
- **Deduplicated Physical Events**: 1,364 unique events
- **Deduplication Strategy**: Spatiotemporal clustering within $1,000\text{ m}$ haversine distance and $\le 2.0\text{ hours}$ orbital separation.

### 1.4 Morocco Administrative Boundaries
- **Source**: GeoBoundaries gbOpen MAR ADM0 (National Boundary)
- **Geometry**: MultiPolygon in EPSG:4326, projected to MODIS Sinusoidal coordinates.

### 1.5 Precipitation (GPM IMERG) Status
- **Source**: NASA GES DISC GPM_3IMERGDF
- **Status**: Access blocked via NASA Earthdata HTTP 403 Forbidden due to external credentials/authorization boundary.
- **Architectural Handling**: Maintained as an optional feature in ingestion modules. Analytical and forecasting models operate on NDVI, LST, and FIRMS without blocking.
