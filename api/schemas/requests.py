from pydantic import BaseModel, Field
from typing import Optional

class RegionRequest(BaseModel):
    region_id: str = Field(default="morocco", description="Target region identifier")
    start_date: Optional[str] = Field(default=None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(default=None, description="End date in YYYY-MM-DD format")

class TrendRequest(BaseModel):
    indicator: str = Field(default="NDVI", description="Target indicator: 'NDVI' or 'LST'")
    start_date: Optional[str] = Field(default=None, description="Optional start filter YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="Optional end filter YYYY-MM-DD")

class AnomalyRequest(BaseModel):
    indicator: str = Field(default="NDVI", description="Target indicator: 'NDVI' or 'LST'")
    threshold_sigma: float = Field(default=2.0, ge=1.0, le=4.0, description="Standard deviation threshold")

class ForecastRequest(BaseModel):
    region_id: str = Field(default="morocco", description="Target region identifier")
    indicator: str = Field(default="NDVI", description="Indicator: 'NDVI' or 'LST'")
    model: str = Field(default="random_forest", description="'random_forest' or 'seasonal_naive'")
    steps: int = Field(default=23, ge=1, le=46, description="Number of future 16-day composite steps to forecast (23 = 1 year)")

class WildfireFilterRequest(BaseModel):
    min_frp: float = Field(default=0.0, ge=0.0, description="Minimum Fire Radiative Power (MW)")
    start_date: Optional[str] = Field(default=None, description="Start acquisition date YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="End acquisition date YYYY-MM-DD")
    limit: int = Field(default=100, ge=1, le=1500, description="Maximum number of events to return")
