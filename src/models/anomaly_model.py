"""
Unsupervised Anomaly Detection Model:
Isolation Forest-based abnormality scoring trained on customer behavioral baselines and velocity features.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.configs.settings import MODEL_FEATURES, MODELS_DIR


class UnsupervisedAnomalyModel:
    """Isolation Forest anomaly detector outputting normalized abnormality scores in [0.0, 1.0]."""

    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.features: List[str] = [
            f for f in MODEL_FEATURES
            if f not in ["connected_case_count", "suspicious_neighbor_count"]
        ]
        self.min_score: float = -0.5
        self.max_score: float = 0.5

    def fit(self, X: pd.DataFrame):
        """Fits the Isolation Forest on normal/unsupervised feature representations."""
        X_aligned = X[self.features].fillna(0.0)
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_aligned)

        # Calibrate score range on training sample
        raw_scores = self.model.decision_function(X_aligned)
        self.min_score = float(np.percentile(raw_scores, 1))
        self.max_score = float(np.percentile(raw_scores, 99))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns normalized anomaly score in [0.0, 1.0] where 1.0 represents extreme abnormality.
        Isolation forest decision_function returns negative values for outliers and positive for inliers.
        We invert and scale so higher = more anomalous.
        """
        if self.model is None:
            raise ValueError("Anomaly model has not been fitted or loaded yet.")
        X_aligned = X[self.features].fillna(0.0)
        raw_decision = self.model.decision_function(X_aligned)

        # Invert: lower raw_decision means higher anomaly
        # Normalize roughly between [0, 1]
        denom = max(1e-4, self.max_score - self.min_score)
        normalized = 1.0 - (raw_decision - self.min_score) / denom
        clipped = np.clip(normalized, 0.0, 1.0)
        return np.round(clipped, 4)

    def evaluate_precision_at_k(self, X_eval: pd.DataFrame, y_eval: pd.Series, top_k_pct: float = 0.05) -> Dict[str, Any]:
        """
        Checks how many top unsupervised anomalies are labelled fraud on a held-out labelled subset.
        """
        anomaly_scores = self.score(X_eval)
        y_true = np.array(y_eval)
        n = len(y_true)
        k = max(1, int(n * top_k_pct))
        top_k_indices = np.argsort(anomaly_scores)[-k:]
        hits = int(y_true[top_k_indices].sum())

        return {
            "top_k_count": k,
            "anomaly_precision_at_top_k": round(float(hits / k), 4),
            "total_frauds_captured": hits,
            "total_frauds_in_eval": int(y_true.sum())
        }

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Saves model to disk."""
        target_path = filepath or (MODELS_DIR / "anomaly_isolation_forest.joblib")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "features": self.features,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "contamination": self.contamination
        }, target_path)
        return target_path

    def load(self, filepath: Optional[Path] = None) -> bool:
        """Loads model from disk."""
        target_path = filepath or (MODELS_DIR / "anomaly_isolation_forest.joblib")
        if not target_path.exists():
            return False
        data = joblib.load(target_path)
        self.model = data["model"]
        self.features = data["features"]
        self.min_score = data["min_score"]
        self.max_score = data["max_score"]
        self.contamination = data.get("contamination", self.contamination)
        return True
