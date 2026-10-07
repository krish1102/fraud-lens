"""
Risk Engine:
Combines supervised fraud probability, unsupervised anomaly score, and graph network risk
using project-configured weights. Assigns risk bands and extracts explanatory signals.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.configs.settings import DEFAULT_RISK_WEIGHTS, RISK_BANDS


class RiskEngine:
    """Combines multi-modal risk evidence into an actionable investigation priority score."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or dict(DEFAULT_RISK_WEIGHTS)
        # Normalize weights so they sum to 1.0
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

    def calculate_combined_score(
        self,
        fraud_probability: float,
        anomaly_score: float,
        network_risk: float
    ) -> float:
        """
        Calculates heuristic combined investigation priority score.
        Formula: w_fraud * p_fraud + w_anomaly * score_anomaly + w_network * score_network
        """
        score = (
            self.weights.get("fraud_probability", 0.50) * fraud_probability +
            self.weights.get("anomaly_score", 0.30) * anomaly_score +
            self.weights.get("network_risk", 0.20) * network_risk
        )
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    @staticmethod
    def get_risk_band(score: float) -> str:
        """Categorizes combined risk score into HIGH, MEDIUM, or LOW band."""
        if score >= RISK_BANDS["HIGH"]:
            return "HIGH"
        elif score >= RISK_BANDS["MEDIUM"]:
            return "MEDIUM"
        else:
            return "LOW"

    @staticmethod
    def extract_detected_signals(row: pd.Series) -> List[str]:
        """
        Inspects row attributes and extracts human-readable behavioral and network risk triggers.
        """
        signals = []

        # Amount signals
        amt_ratio = row.get("amount_vs_customer_avg", 1.0)
        zscore = row.get("customer_amount_zscore", 0.0)
        if amt_ratio >= 3.0:
            signals.append(f"Amount {amt_ratio:.1f}x higher than customer average baseline")
        elif amt_ratio >= 2.0:
            signals.append(f"Amount elevated ({amt_ratio:.1f}x customer average)")
        if zscore >= 2.5:
            signals.append(f"Statistical outlier amount (z-score: +{zscore:.1f})")

        # Velocity signals
        v5 = row.get("transactions_last_5m", 0)
        v30 = row.get("transactions_last_30m", 0)
        if v5 >= 2:
            signals.append(f"High immediate burst velocity ({int(v5)} txns in last 5m)")
        elif v30 >= 3:
            signals.append(f"Elevated velocity window ({int(v30)} txns in last 30m)")

        # Inter-transaction gap
        gap = row.get("time_since_previous_txn", -1)
        if 0 <= gap <= 60:
            signals.append(f"Rapid-fire transaction ({int(gap)}s since previous transaction)")

        # Novelty signals
        if row.get("new_device", 0) == 1 and row.get("customer_transaction_count", 0) > 0:
            signals.append("New/unrecognized hardware device fingerprint")
        if row.get("new_ip", 0) == 1 and row.get("customer_transaction_count", 0) > 0:
            signals.append("Transaction routed through unfamiliar IP address")
        if row.get("new_merchant", 0) == 1 and row.get("customer_transaction_count", 0) > 0:
            signals.append("First-time transaction with this merchant")

        # Temporal signals
        if row.get("is_night", 0) == 1:
            hr_dev = row.get("hour_deviation_from_customer_norm", 0)
            if hr_dev >= 4:
                signals.append(f"Unusual transaction hour (night time, +{int(hr_dev)}h from usual pattern)")

        # Network signals
        shared_dev = row.get("device_customer_count", 1)
        shared_ip = row.get("ip_customer_count", 1)
        suspicious_neigh = row.get("suspicious_neighbor_count", 0)

        if shared_dev > 2:
            signals.append(f"Shared device infrastructure ({int(shared_dev)} distinct customers on device)")
        if shared_ip > 3:
            signals.append(f"Shared IP hub ({int(shared_ip)} distinct customers on IP)")
        if suspicious_neigh > 0:
            signals.append(f"Linked to {int(suspicious_neigh)} previously flagged suspicious entities")

        if not signals:
            signals.append("Transaction within typical historical behavioral parameters")

        return signals
