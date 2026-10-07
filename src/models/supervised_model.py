"""
Supervised Fraud Prediction Model:
LightGBM / XGBoost-based classifier handling extreme class imbalance and tabular feature interactions.
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)
from lightgbm import LGBMClassifier
import xgboost as xgb
from src.configs.settings import MODEL_FEATURES, MODEL_VERSION, MODELS_DIR


class SupervisedFraudModel:
    """Supervised gradient-boosted fraud classifier with calibrated probabilities and evaluation metrics."""

    def __init__(self, model_type: str = "lightgbm", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.features: List[str] = list(MODEL_FEATURES)
        self.model_version = MODEL_VERSION
        self.metrics: Dict[str, Any] = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Trains the gradient boosted tree classifier with class imbalance weighting.
        """
        # Align features
        X_tr = X_train[self.features].fillna(0.0)
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        scale_pos = max(1.0, float(neg_count / max(1, pos_count)))

        if self.model_type == "lightgbm":
            self.model = LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                scale_pos_weight=scale_pos,
                random_state=self.random_state,
                verbose=-1
            )
            self.model.fit(X_tr, y_train)
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                scale_pos_weight=scale_pos,
                random_state=self.random_state,
                eval_metric="logloss"
            )
            self.model.fit(X_tr, y_train)

        # Evaluate if validation set provided
        if X_val is not None and y_val is not None:
            self.metrics = self.evaluate(X_val, y_val)
        else:
            self.metrics = self.evaluate(X_train, y_train)

        return self.metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns fraud probability vector [P(y=1)] for input feature matrix.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        X_aligned = X[self.features].fillna(0.0)
        probs = self.model.predict_proba(X_aligned)[:, 1]
        return np.round(probs, 4)

    def evaluate(self, X_eval: pd.DataFrame, y_eval: pd.Series, top_k_pct: float = 0.05) -> Dict[str, Any]:
        """
        Computes fraud-appropriate metrics: PR-AUC, ROC-AUC, Precision@Top-k, Recall@Top-k, F1, Brier score.
        """
        probs = self.predict_proba(X_eval)
        y_true = np.array(y_eval)

        # Precision-Recall AUC
        precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
        pr_auc = auc(recalls, precisions)

        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_true, probs)
        except Exception:
            roc_auc = 0.5

        # Top-k metrics (analyst review capacity simulation)
        n_eval = len(y_true)
        k = max(1, int(n_eval * top_k_pct))
        top_k_indices = np.argsort(probs)[-k:]
        top_k_actual_frauds = y_true[top_k_indices].sum()
        total_frauds = max(1, y_true.sum())

        precision_at_k = round(float(top_k_actual_frauds / k), 4)
        recall_at_k = round(float(top_k_actual_frauds / total_frauds), 4)

        # Standard threshold metrics (at p >= 0.5)
        preds_50 = (probs >= 0.5).astype(int)
        f1 = round(float(f1_score(y_true, preds_50, zero_division=0)), 4)
        brier = round(float(brier_score_loss(y_true, probs)), 4)

        eval_summary = {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "pr_auc": round(float(pr_auc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "precision_at_top_5pct": precision_at_k,
            "recall_at_top_5pct": recall_at_k,
            "f1_score": f1,
            "brier_score": brier,
            "eval_samples": n_eval,
            "fraud_prevalence": round(float(total_frauds / n_eval), 4)
        }
        return eval_summary

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Saves trained model and metadata to disk."""
        target_path = filepath or (MODELS_DIR / "supervised_fraud_model.joblib")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "model_type": self.model_type,
            "features": self.features,
            "model_version": self.model_version,
            "metrics": self.metrics
        }, target_path)
        return target_path

    def load(self, filepath: Optional[Path] = None) -> bool:
        """Loads trained model and metadata from disk."""
        target_path = filepath or (MODELS_DIR / "supervised_fraud_model.joblib")
        if not target_path.exists():
            return False
        data = joblib.load(target_path)
        self.model = data["model"]
        self.model_type = data["model_type"]
        self.features = data["features"]
        self.model_version = data.get("model_version", self.model_version)
        self.metrics = data.get("metrics", {})
        return True
