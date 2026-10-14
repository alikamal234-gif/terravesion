import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data

def test_regions_overview():
    response = client.get("/regions/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["region_id"] == "morocco"
    assert data["total_observations"] == 271
    assert "bounding_box" in data
    assert "summary_statistics" in data
    assert data["latest_ndvi"] > 0

def test_trends_ndvi():
    response = client.get("/trends?indicator=NDVI")
    assert response.status_code == 200
    data = response.json()
    assert data["indicator"] == "NDVI"
    assert "sen_slope" in data
    assert "p_value" in data
    assert len(data["time_series"]) == 271

def test_trends_lst():
    response = client.get("/trends?indicator=LST")
    assert response.status_code == 200
    data = response.json()
    assert data["indicator"] == "LST"
    assert len(data["time_series"]) == 271

def test_trends_invalid():
    response = client.get("/trends?indicator=INVALID")
    assert response.status_code == 400

def test_anomalies_ndvi():
    response = client.get("/anomalies?indicator=NDVI&threshold_sigma=2.0")
    assert response.status_code == 200
    data = response.json()
    assert data["indicator"] == "NDVI"
    assert data["total_observations"] == 271
    assert "total_anomalies" in data
    assert len(data["anomalies"]) == 271

def test_anomalies_isolation_forest():
    response = client.get("/anomalies?indicator=NDVI&method=isolation_forest")
    assert response.status_code == 200
    data = response.json()
    assert data["indicator"] == "NDVI"
    assert "Isolation Forest" in data["methodology"]
    assert data["total_anomalies"] > 0
    assert len(data["anomalies"]) == 271

def test_forecasts():
    response = client.get("/forecasts?region=morocco&indicator=NDVI&steps=23")
    assert response.status_code == 200
    data = response.json()
    assert data["region"] == "morocco"
    assert data["indicator"] == "NDVI"
    assert data["forecast_steps"] == 23
    assert len(data["forecast"]) == 23
    assert "model_metrics" in data
    assert "baseline_metrics" in data
    # Random Forest should have high R2 and low RMSE
    assert data["model_metrics"]["rmse"] < data["baseline_metrics"]["rmse"]
    assert "feature_importances" in data
    assert len(data["feature_importances"]) > 0

def test_forecast_post_predict_random_forest():
    payload = {
        "region_id": "morocco",
        "indicator": "NDVI",
        "model": "random_forest",
        "steps": 12
    }
    response = client.post("/forecasts/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "Random_Forest_Regressor"
    assert data["forecast_steps"] == 12
    assert len(data["forecast"]) == 12
    assert data["feature_importances"] is not None
    assert len(data["feature_importances"]) == 12
    assert data["limitations"] is not None
    assert data["training_period"] is not None

def test_forecast_post_predict_seasonal_naive():
    payload = {
        "region_id": "morocco",
        "indicator": "NDVI",
        "model": "seasonal_naive",
        "steps": 6
    }
    response = client.post("/forecasts/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "Seasonal_Naive_Baseline"
    assert data["forecast_steps"] == 6
    assert len(data["forecast"]) == 6

def test_forecast_invalid_inputs():
    # Unsupported indicator LST
    resp1 = client.post("/forecasts/predict", json={"region_id": "morocco", "indicator": "LST", "steps": 23})
    assert resp1.status_code == 400
    assert "Only 'NDVI' models are trained" in resp1.json()["detail"]

    # Unsupported geographic region
    resp2 = client.post("/forecasts/predict", json={"region_id": "algeria", "indicator": "NDVI", "steps": 23})
    assert resp2.status_code == 400
    assert "Geographic scope 'algeria' is not supported" in resp2.json()["detail"]

    # Unsupported model
    resp3 = client.post("/forecasts/predict", json={"region_id": "morocco", "indicator": "NDVI", "model": "lstm", "steps": 23})
    assert resp3.status_code == 400
    assert "Model 'lstm' is not supported" in resp3.json()["detail"]

    # Unsupported horizon
    resp4 = client.post("/forecasts/predict", json={"region_id": "morocco", "indicator": "NDVI", "steps": 100})
    assert resp4.status_code == 422 or resp4.status_code == 400

def test_wildfire():
    response = client.get("/wildfire?limit=25")
    assert response.status_code == 200
    data = response.json()
    assert data["total_events"] == 1364
    assert len(data["events"]) == 25
    assert data["max_frp"] > 100.0

def test_clustering():
    response = client.get("/clustering")
    assert response.status_code == 200
    data = response.json()
    assert data["n_clusters"] >= 2
    assert data["silhouette_score"] > 0.4
    assert len(data["clusters"]) >= 2
