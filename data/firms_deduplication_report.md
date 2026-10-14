# NASA FIRMS Ingestion and Spatiotemporal Deduplication Report

## 1. Input Data Discovery & Provenance
- **Raw Files Ingested**:
  - `data/raw/firms\DL_FIRE_J1V-C2_818992\fire_nrt_J1V-C2_818992.csv`
  - `data/raw/firms\DL_FIRE_J2V-C2_818993\fire_nrt_J2V-C2_818993.csv`
  - `data/raw/firms\DL_FIRE_SV-C2_818994\fire_nrt_SV-C2_818994.csv`
- **Instruments**: VIIRS (375 m active fire detection)
- **Satellites**: NOAA-20 (J1V), NOAA-21 (J2V), Suomi-NPP (SV)
- **Coverage Period**: 2026-07-01 to 2026-10-07
- **Raw Detections Ingested**: 3,075 records

## 2. Deduplication Methodology
- **Physical Context**: The polar-orbiting VIIRS constellation features overlapping swaths with ~50-100 minute orbital time separation. A single ongoing fire event on the ground is routinely captured multiple times across successive overpasses.
- **Spatial Threshold**: $1,000\text{ m}$ ($1\text{ km}$ haversine distance), matching the footprint of our analytical environmental grid.
- **Temporal Threshold**: $\le 2.0\text{ hours}$ window, linking consecutive satellite overpasses while separating distinct diurnal burn cycles.
- **Aggregation Rules**:
  - Event location: Centroid of detection points.
  - Timestamp: First and last observed UTC timestamps.
  - Confidence: Maximum confidence among constituent detections.
  - Fire Radiative Power (FRP): Total radiative power summed (`frp_total_mw`) and peak intensity recorded (`frp_max_mw`).
  - Traceability: List of observing satellites and detection count (`n_detections`) preserved.

## 3. Deduplication Results
| Metric | Value |
|---|---|
| **Raw Input Observations** | 3,075 |
| **Unique Physical Fire Events** | 1,364 |
| **Multi-Detection / Duplicate Points Grouped** | 1,711 (55.64%) |
| **Multi-Satellite Verified Events** | 440 (32.3%) |
| **Peak Event FRP** | 490.17 MW |
| **Output File** | `data/interim/firms_morocco_deduplicated.csv` |

## 4. Scientific Caveats & Limitations
1. **Hotspots $\ne$ Burned Area**: Active fire hotspots represent thermal anomaly pixel centroids at satellite overpass time. They do not constitute exact fire perimeters, burned area polygons, or fire duration.
2. **Detection Omission**: Cloud cover, dense smoke, and rapid fires that ignite and extinguish between overpass windows may not be captured.
