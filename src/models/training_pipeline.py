"""
Training Pipeline:
Separates model training from inference.
Trains Supervised Fraud Model (LightGBM) and Unsupervised Anomaly Model (Isolation Forest)
on development/benchmark data, evaluates performance, and saves versioned model artifacts.
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.configs.settings import (
    MODELS_DIR,
    MODEL_FEATURES,
    FEATURE_VERSION,
    MODEL_VERSION
)
from src.features.time_aware_engine import TimeAwareFeatureEngine
from src.models.supervised_model import SupervisedFraudModel
from src.models.anomaly_model import UnsupervisedAnomalyModel


class TrainingPipeline:
    """Orchestrates model training, evaluation, and artifact serialization."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.feature_engine = TimeAwareFeatureEngine()
        self.supervised_model = SupervisedFraudModel(model_type="lightgbm")
        self.anomaly_model = UnsupervisedAnomalyModel()

    def generate_synthetic_development_data(self, n_rows: int = 2500, random_seed: int = 42) -> pd.DataFrame:
        """
        Synthesizes a realistic IEEE-CIS-style e-commerce development dataset
        with true underlying fraud patterns (ATO, Card Testing, Velocity Spikes, Mule Hubs).
        """
        np.random.seed(random_seed)
        start_time = pd.Timestamp("2026-09-01 00:00:00")

        # 120 analytical customers
        customers = [f"C{str(i).zfill(5)}" for i in range(1, 121)]
        devices = [f"D{str(i).zfill(4)}" for i in range(1, 80)]
        ips = [f"192.168.{i // 256}.{i % 256}" for i in range(1, 150)]
        merchants = [f"M{str(i).zfill(4)}" for i in range(1, 50)]
        payment_methods = ["card", "upi", "netbanking", "wallet"]

        # Ensure every customer has a usable history, then vary activity levels.
        baseline_repeats = min(3, n_rows // len(customers))
        customer_sequence = np.repeat(customers, baseline_repeats).tolist()
        remaining_rows = n_rows - len(customer_sequence)
        if remaining_rows:
            activity_weights = np.random.lognormal(mean=0.0, sigma=0.8, size=len(customers))
            activity_probabilities = activity_weights / activity_weights.sum()
            customer_sequence.extend(
                np.random.choice(customers, size=remaining_rows, p=activity_probabilities).tolist()
            )
        np.random.shuffle(customer_sequence)

        # Assign preferred baseline devices & merchants to customers
        cust_pref = {}
        for c in customers:
            cust_pref[c] = {
                "base_dev": np.random.choice(devices[:50]),
                "base_ip": np.random.choice(ips[:100]),
                "base_merchants": np.random.choice(merchants, size=3, replace=False).tolist(),
                "base_amount_mean": float(np.random.uniform(25.0, 150.0)),
                "base_amount_std": float(np.random.uniform(5.0, 30.0))
            }

        rows = []
        curr_time = start_time

        # Target fraud syndicate hubs
        syndicate_device = "D9999"
        syndicate_ip = "198.51.100.99"

        for i in range(1, n_rows + 1):
            txn_id = f"T{str(i).zfill(6)}"
            cust = customer_sequence[i - 1]
            pref = cust_pref[cust]

            # Step forward in time (0 to 12 minutes normally)
            step_seconds = int(np.random.exponential(scale=300))
            curr_time += pd.Timedelta(seconds=max(5, step_seconds))

            # Determine fraud probability for this synthetic row (~4% fraud prevalence)
            is_fraud_scenario = np.random.rand() < 0.04
            is_fraud = 0

            if not is_fraud_scenario:
                # Normal transaction
                amt = max(5.0, np.random.normal(pref["base_amount_mean"], pref["base_amount_std"]))
                dev = pref["base_dev"] if np.random.rand() < 0.90 else np.random.choice(devices)
                ip = pref["base_ip"] if np.random.rand() < 0.85 else np.random.choice(ips)
                merch = np.random.choice(pref["base_merchants"])
                is_fraud = 0
            else:
                # Inject fraud typology
                scenario_type = np.random.choice(["ATO", "CARD_TESTING", "VELOCITY_BURST", "SYNDICATE"])
                if scenario_type == "ATO":
                    # Huge amount, new device, new IP
                    amt = pref["base_amount_mean"] * np.random.uniform(4.0, 10.0) + 200.0
                    dev = np.random.choice(devices[50:])
                    ip = np.random.choice(ips[100:])
                    merch = np.random.choice(merchants)
                    is_fraud = 1
                elif scenario_type == "CARD_TESTING":
                    # Tiny amount, rapid
                    amt = float(np.random.choice([0.99, 1.50, 2.00, 3.25]))
                    dev = syndicate_device
                    ip = syndicate_ip
                    merch = np.random.choice(merchants)
                    is_fraud = 1
                elif scenario_type == "VELOCITY_BURST":
                    # Step timestamp back to create a micro-burst within 2 minutes of prior
                    curr_time = curr_time - pd.Timedelta(seconds=max(0, step_seconds - 25))
                    amt = pref["base_amount_mean"] * np.random.uniform(2.5, 5.0)
                    dev = np.random.choice(devices)
                    ip = np.random.choice(ips)
                    merch = np.random.choice(merchants)
                    is_fraud = 1
                else:  # SYNDICATE
                    amt = max(50.0, np.random.normal(pref["base_amount_mean"] * 2.0, 50.0))
                    dev = syndicate_device
                    ip = syndicate_ip
                    merch = np.random.choice(merchants)
                    is_fraud = 1

            rows.append({
                "transaction_id": txn_id,
                "customer_id": cust,
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": round(float(amt), 2),
                "device_id": dev,
                "merchant_id": merch,
                "ip_address": ip,
                "payment_method": np.random.choice(payment_methods),
                "isFraud": is_fraud
            })

        df = pd.DataFrame(rows)
        return df

    def train_and_persist(
        self,
        training_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Runs complete training lifecycle:
          1. Time-aware feature engineering
          2. Train/validation split (time-respecting)
          3. Model training (supervised + unsupervised)
          4. Evaluation metrics calculation
          5. Artifact persistence to disk
        """
        if training_df is None:
            raw_dev_df = self.generate_synthetic_development_data(n_rows=3000)
        else:
            raw_dev_df = training_df.copy()

        # Separate target
        if "isFraud" in raw_dev_df.columns:
            target = raw_dev_df["isFraud"].astype(int)
        elif "is_fraud" in raw_dev_df.columns:
            target = raw_dev_df["is_fraud"].astype(int)
        else:
            raise ValueError("Training dataset must contain 'isFraud' or 'is_fraud' column.")

        # 1. Feature Engineering
        enriched_df, feature_matrix = self.feature_engine.build_features(raw_dev_df)

        # 2. Time-aware 80/20 train-test split (train on earlier, test on later)
        split_idx = int(len(feature_matrix) * 0.80)
        X_train = feature_matrix.iloc[:split_idx]
        y_train = target.iloc[:split_idx]
        X_val = feature_matrix.iloc[split_idx:]
        y_val = target.iloc[split_idx:]

        # 3. Train Supervised Model
        supervised_metrics = self.supervised_model.train(X_train, y_train, X_val, y_val)
        sup_path = self.supervised_model.save(self.models_dir / "supervised_fraud_model.joblib")

        # 4. Train Unsupervised Anomaly Model (fit on normal non-fraud training samples)
        normal_train = X_train[y_train == 0]
        self.anomaly_model.fit(normal_train)
        anomaly_metrics = self.anomaly_model.evaluate_precision_at_k(X_val, y_val)
        anom_path = self.anomaly_model.save(self.models_dir / "anomaly_isolation_forest.joblib")

        # 5. Save Experiment Metadata Manifest
        manifest = {
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "supervised_model_type": "lightgbm",
            "features_list": MODEL_FEATURES,
            "supervised_metrics": supervised_metrics,
            "anomaly_metrics": anomaly_metrics,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "supervised_artifact": str(sup_path.name),
            "anomaly_artifact": str(anom_path.name)
        }

        with open(self.models_dir / "model_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest
