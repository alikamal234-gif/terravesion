import pytest
import pandas as pd
import numpy as np
from src.ingestion.firms_loader import FIRMSLoader

def test_firms_validation_and_deduplication():
    # Synthetic test fixture simulating 4 raw hotspot detections
    # Detections 1 and 2: same physical fire (distance ~100m, time separation 30 mins, different satellites)
    # Detection 3: distinct fire 50 km away
    # Detection 4: out of bounds (invalid latitude)
    raw_data = pd.DataFrame([
        {
            "latitude": 33.5892,
            "longitude": -7.6038,
            "acq_date": "2023-07-15",
            "acq_time": 1330,
            "confidence": "n",
            "frp": 12.5,
            "brightness": 320.1,
            "satellite": "N20",
            "daynight": "D"
        },
        {
            "latitude": 33.5900,  # ~100 meters away
            "longitude": -7.6040,
            "acq_date": "2023-07-15",
            "acq_time": 1400,  # 30 mins later
            "confidence": "h",
            "frp": 15.0,
            "brightness": 325.4,
            "satellite": "N21",
            "daynight": "D"
        },
        {
            "latitude": 34.0208,  # ~50 km away (Rabat region)
            "longitude": -6.8416,
            "acq_date": "2023-07-15",
            "acq_time": 1335,
            "confidence": "n",
            "frp": 8.0,
            "brightness": 315.0,
            "satellite": "N20",
            "daynight": "D"
        },
        {
            "latitude": 45.0,  # Outside Morocco (invalid)
            "longitude": 2.0,
            "acq_date": "2023-07-15",
            "acq_time": 1200,
            "confidence": "l",
            "frp": 5.0,
            "brightness": 300.0,
            "satellite": "SV",
            "daynight": "D"
        }
    ])

    loader = FIRMSLoader()
    
    # Save temporary CSV to test load_and_validate
    temp_csv = "data/raw/firms_temp_test.csv"
    raw_data.to_csv(temp_csv, index=False)
    
    try:
        clean_df = loader.load_and_validate([temp_csv])
        # Detection 4 should be filtered out
        assert len(clean_df) == 3, f"Expected 3 valid detections, got {len(clean_df)}"
        assert clean_df["latitude"].max() <= 37.0
        assert clean_df["latitude"].min() >= 20.0

        # Run spatiotemporal deduplication (1 km, 2 hours)
        dedup_events, clustered_raw = loader.deduplicate_spatiotemporal(clean_df, spatial_radius_m=1000.0, time_window_hours=2.0)
        
        # Detections 1 and 2 should be grouped into 1 event, Detection 3 is another event -> total 2 events
        assert len(dedup_events) == 2, f"Expected 2 deduplicated events, got {len(dedup_events)}"
        
        # Check that cluster with 2 detections captured multi-satellite flag and summed FRP
        clustered_event = dedup_events[dedup_events["n_detections"] == 2].iloc[0]
        assert bool(clustered_event["is_multi_satellite"]) is True
        assert clustered_event["confidence"] == "h"
        assert abs(clustered_event["frp_total_mw"] - 27.5) < 1e-4
    finally:
        import os
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
