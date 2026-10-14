from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from api.routes import regions, trends, anomalies, forecasts, wildfire, clustering

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TerraPulse AI API",
    description="Operational API for Earth Observation analytics, trends, anomalies, forecasting, and wildfire intelligence across Morocco.",
    version="1.0.1"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all route modules
app.include_router(regions.router)
app.include_router(trends.router)
app.include_router(anomalies.router)
app.include_router(forecasts.router)
app.include_router(wildfire.router)
app.include_router(clustering.router)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "TerraPulse AI Operational API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": [
            "/regions/overview",
            "/trends?indicator=NDVI",
            "/anomalies?indicator=NDVI&threshold_sigma=2.0",
            "/forecasts?region=morocco&indicator=NDVI&steps=23",
            "/wildfire?limit=100",
            "/clustering"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
