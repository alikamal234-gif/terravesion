from fastapi import APIRouter, HTTPException
import os
import json
import logging
from api.schemas.responses import ClusteringResponse, ClusterProfile

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/clustering", tags=["Clustering"])

@router.get("", response_model=ClusteringResponse)
async def get_clustering_profiles():
    """
    Returns empirical K-Means environmental condition profiles.
    """
    cluster_file = "data/processed/clustering_profiles.json"
    if not os.path.exists(cluster_file):
        raise HTTPException(status_code=404, detail="Clustering profiles not yet computed.")

    with open(cluster_file, "r") as f:
        data = json.load(f)

    profiles = [
        ClusterProfile(
            cluster_id=item["cluster_id"],
            name=item["name"],
            mean_ndvi=item["mean_ndvi"],
            mean_lst_celsius=item["mean_lst_celsius"],
            description=item["description"],
            estimated_coverage_pct=item["estimated_coverage_pct"]
        )
        for item in data.get("clusters", [])
    ]

    return ClusteringResponse(
        n_clusters=data.get("n_clusters", 3),
        silhouette_score=data.get("silhouette_score", 0.0),
        clusters=profiles,
        interpretation_note=data.get("interpretation_note", "")
    )
