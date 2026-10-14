# TerraPulse AI — Scientific Limitations & Operational Scope

## 1. Indicator Interpretation Boundaries
1. **NDVI $\ne$ Direct Biodiversity**:
   NDVI (Normalized Difference Vegetation Index) quantifies photosynthetic canopy activity, chlorophyll absorption in the red band, and near-infrared reflectance. It is an index of vegetative greenness, biomass density, and drought stress. It must **not** be presented as an unmediated census of botanical diversity, species richness, or fauna abundance.
2. **FIRMS Hotspots $\ne$ Burned Area Perimeters**:
   NASA FIRMS VIIRS detections are instantaneous thermal anomaly radiometry measurements ($375\text{ m}$ pixel centroid). A hotspot indicates an active thermal emission detected at satellite overpass time. It does **not** delineate exact fire perimeters, total burned area hectarage, or distinguish controlled agricultural burns from uncontrolled wildfires without local ancillary data.
3. **Environmental Clusters $\ne$ Ecological Conservation Classes**:
   The K-Means clusters identify empirical regimes of joint vegetation density and thermal irradiance across the annual phenological cycle. They are bioclimatic condition profiles, not statutory nature reserves, forest types, or ecological classification boundaries.

## 2. Sensor and Temporal Resolution
- **MODIS Composite Lag**: NDVI composites span 16 days; LST composites span 8 days. Rapid, short-lived events (e.g., brief flash floods or localized 2-hour burns that extinguish before the next overpass) may not register as acute composite anomalies.
- **Atmospheric & Cloud Contamination**: Despite QC bitmask filtering (MOD13Q1 VI Quality and MOD11A2 QC_Day), severe extended overcast periods during the winter rainy season can reduce valid pixel observations.

## 3. External Precipitation Data Constraint
- GPM IMERG global precipitation retrieval remains blocked at NASA GES DISC due to HTTP 403 authorization restrictions. Precipitation has been documented and maintained as an optional component; analytical and forecasting modules operate autonomously on validated NDVI, LST, and FIRMS observations without blocking.
