import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import xarray as xr
from src.models.clustering.kmeans import EcosystemConditionClustering

logger = logging.getLogger(__name__)

class EnvironmentalClusteringAnalyzer:
    """
    Constructs multi-annual environmental condition profiles from joint NDVI and LST raster dynamics.
    Clusters represent distinct bioclimatic / eco-condition regimes across Morocco.
    """
    @staticmethod
    def extract_raster_features(ds: xr.Dataset, sample_stride: int = 15) -> pd.DataFrame:
        """
        Extracts spatial feature attributes from the processed NetCDF dataset:
        - Mean NDVI, NDVI 90th percentile, NDVI seasonal amplitude
        - Mean LST, LST seasonal amplitude
        - Coordinates (x, y)
        """
        sampled = ds.isel(ydim=slice(0, None, sample_stride), xdim=slice(0, None, sample_stride))
        
        # Temporal aggregations across the multi-year record
        ndvi_mean = sampled["NDVI"].mean(dim="time").values
        ndvi_std = sampled["NDVI"].std(dim="time").values
        ndvi_max = sampled["NDVI"].max(dim="time").values
        ndvi_min = sampled["NDVI"].min(dim="time").values

        lst_mean = sampled["LST"].mean(dim="time").values
        lst_std = sampled["LST"].std(dim="time").values
        lst_max = sampled["LST"].max(dim="time").values
        lst_min = sampled["LST"].min(dim="time").values

        y_vals = sampled.ydim.values
        x_vals = sampled.xdim.values

        records = []
        ny, nx = ndvi_mean.shape
        for iy in range(ny):
            for ix in range(nx):
                nm = ndvi_mean[iy, ix]
                lm = lst_mean[iy, ix]
                # Exclude masked / ocean points
                if np.isnan(nm) or np.isnan(lm):
                    continue

                records.append({
                    "y": float(y_vals[iy]),
                    "x": float(x_vals[ix]),
                    "ndvi_mean": float(nm),
                    "ndvi_amplitude": float(ndvi_max[iy, ix] - ndvi_min[iy, ix]),
                    "ndvi_std": float(ndvi_std[iy, ix]),
                    "lst_mean_k": float(lm),
                    "lst_amplitude_k": float(lst_max[iy, ix] - lst_min[iy, ix]),
                    "lst_std_k": float(lst_std[iy, ix])
                })

        df_feat = pd.DataFrame(records)
        logger.info(f"Extracted {len(df_feat)} valid spatial feature records for environmental clustering.")
        return df_feat

    @staticmethod
    def find_optimal_clusters(X: pd.DataFrame, cluster_range: range = range(2, 6)) -> Tuple[int, Dict[int, float]]:
        """
        Evaluates silhouette scores across candidate cluster counts to select optimal k.
        """
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        scores = {}
        for k in cluster_range:
            model = EcosystemConditionClustering(n_clusters=k, random_state=42)
            model.fit_kmeans(X)
            labels = model.predict_clusters(X)
            score = float(silhouette_score(X_scaled, labels))
            scores[k] = round(score, 4)
            logger.info(f"k={k}: Silhouette Score = {score:.4f}")

        optimal_k = max(scores, key=scores.get)
        logger.info(f"Optimal cluster count selected: k={optimal_k} (score={scores[optimal_k]})")
        return optimal_k, scores

    @staticmethod
    def profile_and_label_clusters(model: EcosystemConditionClustering, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates profiles for each cluster and assigns descriptive bioclimatic labels
        based on empirical vegetative greenness and thermal severity.
        """
        profiles = model.analyze_cluster_profiles(X)
        
        # Qualitative environmental condition labels
        labels = []
        for cluster_id, row in profiles.iterrows():
            ndvi_m = row.get("ndvi_mean", 0)
            lst_m = row.get("lst_mean_k", 300)
            if ndvi_m < 0.12 and lst_m > 308:
                desc = "Hyper-Arid / Desert Ground"
            elif ndvi_m < 0.20:
                desc = "Arid / Semi-Arid Shrubland"
            elif ndvi_m < 0.35:
                desc = "Moderate Canopy / Agro-Pastoral"
            else:
                desc = "Dense Vegetation / Humid Mountain & Forest"
            labels.append(desc)

        profiles["environmental_profile"] = labels
        return profiles
