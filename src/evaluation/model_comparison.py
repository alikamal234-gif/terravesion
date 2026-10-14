import os
import json
import logging
from typing import Dict, List, Any
import pandas as pd

logger = logging.getLogger(__name__)

class ModelComparator:
    """
    Compares time series forecasting models and benchmarks.
    Logs and formats comparative metrics (MAE, RMSE, R2, training time).
    """
    def __init__(self, output_dir: str = "data/processed/evaluation"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def add_result(
        self,
        model_name: str,
        target_variable: str,
        metrics: Dict[str, float],
        training_time_s: float = 0.0,
        notes: str = ""
    ):
        record = {
            "model_name": model_name,
            "target_variable": target_variable,
            "mae": metrics.get("mae", float("nan")),
            "rmse": metrics.get("rmse", float("nan")),
            "r2": metrics.get("r2", float("nan")),
            "training_time_s": round(training_time_s, 3),
            "notes": notes
        }
        self.results.append(record)
        logger.info(f"Added evaluation result for {model_name}: MAE={record['mae']}, RMSE={record['rmse']}, R2={record['r2']}")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def save_summary(self, filename: str = "forecasting_benchmark.json") -> str:
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
        logger.info(f"Saved model comparison to {out_path}")
        return out_path
