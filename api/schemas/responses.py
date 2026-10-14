from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class RegionOverviewResponse(BaseModel):
    region_id: str = "morocco"
    region_name: str = "Morocco (National)"
    bounding_box: Dict[str, float] = Field(..., description="Bounding coordinates min_lon, min_lat, max_lon, max_lat")
    crs: str = "EPSG:4326 / Modis Sinusoidal"
    temporal_coverage: Dict[str, str] = Field(..., description="start_date and end_date")
    total_observations: int
    latest_observation_date: str
    latest_ndvi: float
    latest_lst_celsius: float
    summary_statistics: Dict[str, Any]

class TimePoint(BaseModel):
    timestamp: str
    value: float

class TrendResponse(BaseModel):
    indicator: str
    trend_direction: str
    sen_slope: float
    sen_slope_per_decade: float
    p_value: float
    is_significant: bool
    tau: float
    n_observations: int
    interpretation: str
    time_series: List[TimePoint]

class AnomalyPoint(BaseModel):
    timestamp: str
    actual_value: float
    climatological_mean: float
    anomaly_delta: float
    z_score: float
    is_anomaly: bool
    severity: str

class AnomalySummaryResponse(BaseModel):
    indicator: str
    total_observations: int
    total_anomalies: int
    anomaly_rate_percent: float
    severe_negative_count: int
    severe_positive_count: int
    methodology: str
    anomalies: List[AnomalyPoint]

class ForecastPoint(BaseModel):
    timestamp: str
    predicted_value: float
    lower_bound: float
    upper_bound: float

class ModelMetrics(BaseModel):
    mae: float
    rmse: float
    r2: Optional[float] = None

class FeatureImportance(BaseModel):
    feature_name: str
    importance: float
    description: Optional[str] = None

class ForecastResponse(BaseModel):
    region: str
    indicator: str
    model_type: str
    forecast_steps: int
    forecast: List[ForecastPoint]
    historical: List[TimePoint]
    model_metrics: ModelMetrics
    baseline_metrics: ModelMetrics
    comparison_summary: str
    feature_importances: Optional[List[FeatureImportance]] = None
    generation_timestamp: Optional[str] = None
    training_period: Optional[Dict[str, Any]] = None
    limitations: Optional[List[str]] = None

class WildfireEvent(BaseModel):
    event_id: str
    latitude: float
    longitude: float
    acq_date: str
    acq_time: Optional[str] = None
    satellite: str
    frp: float
    frp_total_mw: Optional[float] = None
    frp_max_mw: Optional[float] = None
    confidence: Optional[str] = None
    pre_ndvi: Optional[float] = None
    post_ndvi: Optional[float] = None
    ndvi_drop: Optional[float] = None

class WildfireResponse(BaseModel):
    total_events: int
    date_range: Dict[str, str]
    mean_frp: float
    max_frp: float
    peak_cluster_frp: Optional[float] = None
    events: List[WildfireEvent]

class ClusterProfile(BaseModel):
    cluster_id: int
    name: str
    mean_ndvi: float
    mean_lst_celsius: float
    description: str
    estimated_coverage_pct: float

class ClusteringResponse(BaseModel):
    n_clusters: int
    silhouette_score: float
    clusters: List[ClusterProfile]
    interpretation_note: str
