import os
import glob
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)

class FIRMSLoader:
    """
    NASA FIRMS active fire detection loader, validator, and spatiotemporal deduplicator.
    Handles VIIRS / MODIS active fire products, filters regional bounding boxes,
    and clusters overlapping cross-satellite detections.
    """
    def __init__(self, raw_dir: str = "data/raw/firms"):
        self.raw_dir = raw_dir

    def discover_raw_files(self) -> List[str]:
        """
        Discovers valid FIRMS CSV files, excluding non-CSV artifacts (e.g. NetCDF)
        and files outside the Morocco geographical area.
        """
        candidate_files = glob.glob(os.path.join(self.raw_dir, "**", "*.csv"), recursive=True)
        valid_files = []
        
        for f in candidate_files:
            try:
                # Read top few rows to validate schema and coordinates
                sample = pd.read_csv(f, nrows=10)
                required_cols = {"latitude", "longitude", "acq_date", "acq_time"}
                if not required_cols.issubset(set(sample.columns)):
                    logger.warning(f"Excluding {f}: missing required columns.")
                    continue
                    
                # Morocco bounding box: Lat [20.0, 36.5], Lon [-18.0, 0.0]
                if sample["longitude"].between(-18.0, 0.0).all() and sample["latitude"].between(20.0, 37.0).all():
                    valid_files.append(f)
                    logger.info(f"Discovered valid Morocco FIRMS CSV: {f}")
                else:
                    logger.info(f"Skipping {f}: Coordinates outside Morocco bounding box.")
            except Exception as e:
                logger.error(f"Error reading {f}: {e}")
                
        return sorted(valid_files)

    def load_and_validate(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Loads and validates raw FIRMS CSV records across all input files.
        Standardizes column names, parses UTC timestamps, and validates ranges.
        """
        if not file_paths:
            raise FileNotFoundError("No valid FIRMS CSV files provided.")

        dfs = []
        for path in file_paths:
            df = pd.read_csv(path)
            df["source_file"] = os.path.basename(path)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        initial_count = len(combined)

        # Standardize brightness column names if needed
        if "bright_ti4" in combined.columns and "brightness" not in combined.columns:
            combined["brightness"] = combined["bright_ti4"]
        if "bright_t31" not in combined.columns and "bright_ti5" in combined.columns:
            combined["bright_t31"] = combined["bright_ti5"]

        # Parse acq_date and acq_time (HHMM format)
        combined["acq_time_str"] = combined["acq_time"].astype(str).str.zfill(4)
        combined["datetime_utc"] = pd.to_datetime(
            combined["acq_date"] + " " + combined["acq_time_str"].str[:2] + ":" + combined["acq_time_str"].str[2:],
            utc=True,
            errors="coerce"
        )

        # Coordinate and range validation
        valid_mask = (
            combined["latitude"].between(20.0, 37.0) &
            combined["longitude"].between(-18.0, 0.0) &
            combined["datetime_utc"].notnull()
        )
        
        # FRP validation (FRP must be non-negative)
        if "frp" in combined.columns:
            valid_mask = valid_mask & (combined["frp"] >= 0.0)

        clean_df = combined[valid_mask].copy().sort_values("datetime_utc").reset_index(drop=True)
        logger.info(f"Validated {len(clean_df)} valid detections from {initial_count} raw records.")
        return clean_df

    def deduplicate_spatiotemporal(
        self,
        df: pd.DataFrame,
        spatial_radius_m: float = 1000.0,
        time_window_hours: float = 2.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs spatiotemporal clustering to deduplicate active fire hotspots.
        Groups observations within spatial_radius_m (default 1 km) and time_window_hours (default 2h,
        matching orbital overpass gap across constellation satellites).

        Returns:
            dedup_events: One row per physical fire event cluster with aggregated metrics.
            clustered_raw: The original observations labeled with their assigned event_id.
        """
        if df.empty:
            raise ValueError("Input dataframe is empty.")

        df_sorted = df.sort_values("datetime_utc").reset_index(drop=True)
        n = len(df_sorted)

        # Convert coordinates to radians for BallTree haversine metric
        earth_radius_m = 6371000.0
        radius_rad = spatial_radius_m / earth_radius_m
        coords_rad = np.radians(df_sorted[["latitude", "longitude"]].values)
        tree = BallTree(coords_rad, metric="haversine")

        cluster_ids = np.full(n, -1, dtype=int)
        current_cluster = 0
        window_seconds = time_window_hours * 3600.0

        for i in range(n):
            if cluster_ids[i] != -1:
                continue

            t_i = df_sorted.loc[i, "datetime_utc"]
            neighbors = tree.query_radius(coords_rad[i:i+1], r=radius_rad)[0]

            # Match neighbors that are unassigned and within the temporal window
            time_diffs = np.abs((df_sorted.loc[neighbors, "datetime_utc"] - t_i).dt.total_seconds().values)
            matching_neighbors = neighbors[(cluster_ids[neighbors] == -1) & (time_diffs <= window_seconds)]

            for m in matching_neighbors:
                cluster_ids[m] = current_cluster

            current_cluster += 1

        df_sorted["event_id"] = cluster_ids

        # Build aggregated event records
        confidence_rank = {"l": 1, "n": 2, "h": 3}
        reverse_conf = {1: "l", 2: "n", 3: "h"}

        event_records = []
        for event_id, group in df_sorted.groupby("event_id"):
            # Centroid
            mean_lat = float(group["latitude"].mean())
            mean_lon = float(group["longitude"].mean())
            t_start = group["datetime_utc"].min()
            t_end = group["datetime_utc"].max()

            # Max confidence
            max_conf_num = group["confidence"].map(confidence_rank).fillna(1).max()
            best_confidence = reverse_conf.get(int(max_conf_num), "n")

            # FRP and Brightness
            frp_sum = float(group["frp"].sum()) if "frp" in group.columns else 0.0
            frp_max = float(group["frp"].max()) if "frp" in group.columns else 0.0
            bright_max = float(group["brightness"].max()) if "brightness" in group.columns else 0.0

            # Satellites involved
            satellites = sorted(list(group["satellite"].unique()))
            is_multi_sat = len(satellites) > 1

            event_records.append({
                "event_id": event_id,
                "latitude": round(mean_lat, 5),
                "longitude": round(mean_lon, 5),
                "datetime_first_utc": t_start.isoformat(),
                "datetime_last_utc": t_end.isoformat(),
                "acq_date": str(t_start.date()),
                "confidence": best_confidence,
                "frp_total_mw": round(frp_sum, 2),
                "frp_max_mw": round(frp_max, 2),
                "brightness_max_k": round(bright_max, 2),
                "n_detections": len(group),
                "satellites": ",".join(satellites),
                "is_multi_satellite": is_multi_sat,
                "daynight": group["daynight"].iloc[0] if "daynight" in group.columns else "U"
            })

        dedup_events = pd.DataFrame(event_records)
        logger.info(
            f"Deduplication complete: {len(df_sorted)} raw observations -> {len(dedup_events)} unique fire events "
            f"({len(df_sorted) - len(dedup_events)} duplicates grouped)."
        )
        return dedup_events, df_sorted

    def process_and_save(
        self,
        output_csv: str = "data/interim/firms_morocco_deduplicated.csv",
        report_md: str = "data/firms_deduplication_report.md"
    ) -> pd.DataFrame:
        """
        Executes end-to-end FIRMS discovery, validation, spatiotemporal deduplication,
        saving the clean interim table and Markdown audit report.
        """
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        files = self.discover_raw_files()
        raw_df = self.load_and_validate(files)
        dedup_events, clustered_raw = self.deduplicate_spatiotemporal(raw_df)

        # Save interim clean table
        dedup_events.to_csv(output_csv, index=False)
        logger.info(f"Saved deduplicated FIRMS events to {output_csv}")

        # Generate report
        report_text = f"""# NASA FIRMS Ingestion and Spatiotemporal Deduplication Report

## 1. Input Data Discovery & Provenance
- **Raw Files Ingested**:
{chr(10).join([f"  - `{f}`" for f in files])}
- **Instruments**: VIIRS (375 m active fire detection)
- **Satellites**: NOAA-20 (J1V), NOAA-21 (J2V), Suomi-NPP (SV)
- **Coverage Period**: {raw_df['datetime_utc'].min().strftime('%Y-%m-%d')} to {raw_df['datetime_utc'].max().strftime('%Y-%m-%d')}
- **Raw Detections Ingested**: {len(raw_df):,} records

## 2. Deduplication Methodology
- **Physical Context**: The polar-orbiting VIIRS constellation features overlapping swaths with ~50-100 minute orbital time separation. A single ongoing fire event on the ground is routinely captured multiple times across successive overpasses.
- **Spatial Threshold**: $1,000\\text{{ m}}$ ($1\\text{{ km}}$ haversine distance), matching the footprint of our analytical environmental grid.
- **Temporal Threshold**: $\\le 2.0\\text{{ hours}}$ window, linking consecutive satellite overpasses while separating distinct diurnal burn cycles.
- **Aggregation Rules**:
  - Event location: Centroid of detection points.
  - Timestamp: First and last observed UTC timestamps.
  - Confidence: Maximum confidence among constituent detections.
  - Fire Radiative Power (FRP): Total radiative power summed (`frp_total_mw`) and peak intensity recorded (`frp_max_mw`).
  - Traceability: List of observing satellites and detection count (`n_detections`) preserved.

## 3. Deduplication Results
| Metric | Value |
|---|---|
| **Raw Input Observations** | {len(raw_df):,} |
| **Unique Physical Fire Events** | {len(dedup_events):,} |
| **Multi-Detection / Duplicate Points Grouped** | {len(raw_df) - len(dedup_events):,} ({((len(raw_df) - len(dedup_events)) / len(raw_df)) * 100:.2f}%) |
| **Multi-Satellite Verified Events** | {int(dedup_events['is_multi_satellite'].sum()):,} ({float(dedup_events['is_multi_satellite'].mean() * 100):.1f}%) |
| **Peak Event FRP** | {dedup_events['frp_max_mw'].max():.2f} MW |
| **Output File** | `{output_csv}` |

## 4. Scientific Caveats & Limitations
1. **Hotspots $\\ne$ Burned Area**: Active fire hotspots represent thermal anomaly pixel centroids at satellite overpass time. They do not constitute exact fire perimeters, burned area polygons, or fire duration.
2. **Detection Omission**: Cloud cover, dense smoke, and rapid fires that ignite and extinguish between overpass windows may not be captured.
"""
        with open(report_md, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info(f"Wrote deduplication report to {report_md}")

        return dedup_events
