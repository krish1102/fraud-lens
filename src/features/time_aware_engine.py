"""
Time-Aware Feature Engineering Engine:
Strictly prevents temporal leakage by sorting on timestamp and computing all customer baseline,
velocity, novelty, and network features using ONLY transactions occurring prior to the current transaction.
"""

from typing import List, Dict, Set, Optional, Tuple
import pandas as pd
import numpy as np
from src.configs.settings import MODEL_FEATURES, FEATURE_VERSION


class TimeAwareFeatureEngine:
    """Engineers time-aware customer baseline, velocity, novelty, and network features."""

    def __init__(self, feature_version: str = FEATURE_VERSION):
        self.feature_version = feature_version

    def build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Takes validated canonical DataFrame and returns:
          1. enriched_df: DataFrame with original canonical fields plus all engineered features
          2. feature_matrix: Exact feature matrix formatted for ML models (reproducible column order)
        """
        # Ensure copy and sort strictly by timestamp
        data = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])

        data = data.sort_values(by=["timestamp", "transaction_id"]).reset_index(drop=True)

        # 1. Transaction-level static features
        data["transaction_amount"] = data["amount"].astype(float)
        data["log_transaction_amount"] = np.log1p(np.maximum(0.0, data["transaction_amount"]))
        data["hour"] = data["timestamp"].dt.hour
        data["day_of_week"] = data["timestamp"].dt.dayofweek
        data["is_night"] = data["hour"].apply(lambda h: 1 if h in [22, 23, 0, 1, 2, 3, 4, 5] else 0)
        data["is_weekend"] = data["day_of_week"].apply(lambda d: 1 if d in [5, 6] else 0)

        # Ensure optional fields exist as strings or empty
        for opt in ["device_id", "merchant_id", "ip_address", "beneficiary_id", "location"]:
            if opt not in data.columns:
                data[opt] = ""
            else:
                data[opt] = data[opt].fillna("").astype(str)

        n = len(data)

        # Preallocate numpy arrays for time-aware features
        customer_avg_amount = np.zeros(n, dtype=float)
        customer_median_amount = np.zeros(n, dtype=float)
        amount_vs_customer_avg = np.ones(n, dtype=float)
        amount_vs_customer_median = np.ones(n, dtype=float)
        customer_amount_zscore = np.zeros(n, dtype=float)
        customer_transaction_count = np.zeros(n, dtype=int)
        unique_device_count = np.zeros(n, dtype=int)
        unique_merchant_count = np.zeros(n, dtype=int)
        hour_deviation_from_customer_norm = np.zeros(n, dtype=float)

        transactions_last_5m = np.zeros(n, dtype=int)
        transactions_last_30m = np.zeros(n, dtype=int)
        amount_last_30m = np.zeros(n, dtype=float)
        transactions_last_24h = np.zeros(n, dtype=int)
        time_since_previous_txn = np.full(n, -1.0, dtype=float)

        new_device = np.zeros(n, dtype=int)
        new_merchant = np.zeros(n, dtype=int)
        new_ip = np.zeros(n, dtype=int)

        device_customer_count = np.ones(n, dtype=int)
        ip_customer_count = np.ones(n, dtype=int)
        entity_degree = np.ones(n, dtype=int)
        suspicious_neighbor_count = np.zeros(n, dtype=int)
        connected_case_count = np.zeros(n, dtype=int)

        # Cumulative trackers per customer
        customer_amounts: Dict[str, List[float]] = {}
        customer_timestamps: Dict[str, List[pd.Timestamp]] = {}
        customer_devices: Dict[str, Set[str]] = {}
        customer_merchants: Dict[str, Set[str]] = {}
        customer_ips: Dict[str, Set[str]] = {}
        customer_hours: Dict[str, List[int]] = {}

        # Entity trackers for graph metrics
        device_to_customers: Dict[str, Set[str]] = {}
        ip_to_customers: Dict[str, Set[str]] = {}

        # Row-by-row strictly historical simulation
        for i in range(n):
            cust = data.at[i, "customer_id"]
            ts = data.at[i, "timestamp"]
            amt = data.at[i, "transaction_amount"]
            hr = data.at[i, "hour"]
            dev = data.at[i, "device_id"]
            merch = data.at[i, "merchant_id"]
            ip = data.at[i, "ip_address"]

            # 1. Evaluate against history prior to this transaction
            if cust in customer_amounts and len(customer_amounts[cust]) > 0:
                past_amts = customer_amounts[cust]
                past_ts = customer_timestamps[cust]
                past_devs = customer_devices[cust]
                past_merchs = customer_merchants[cust]
                past_ips = customer_ips[cust]
                past_hrs = customer_hours[cust]

                # Baselines
                mean_amt = float(np.mean(past_amts))
                med_amt = float(np.median(past_amts))
                std_amt = float(np.std(past_amts)) if len(past_amts) > 1 else 0.0

                customer_avg_amount[i] = mean_amt
                customer_median_amount[i] = med_amt
                amount_vs_customer_avg[i] = round(amt / (mean_amt + 1e-4), 3)
                amount_vs_customer_median[i] = round(amt / (med_amt + 1e-4), 3)
                customer_amount_zscore[i] = round((amt - mean_amt) / (std_amt + 1e-4), 3) if std_amt > 0 else 0.0
                customer_transaction_count[i] = len(past_amts)
                unique_device_count[i] = len(past_devs)
                unique_merchant_count[i] = len(past_merchs)

                # Usual transaction hour (mode)
                usual_hr = max(set(past_hrs), key=past_hrs.count)
                hour_deviation_from_customer_norm[i] = abs(hr - usual_hr)

                # Time since previous transaction (seconds)
                last_ts = past_ts[-1]
                delta_sec = max(0.0, (ts - last_ts).total_seconds())
                time_since_previous_txn[i] = round(delta_sec, 1)

                # Rolling velocity counts strictly prior
                # 5 min window (300 sec)
                t_minus_5m = ts - pd.Timedelta(minutes=5)
                # 30 min window (1800 sec)
                t_minus_30m = ts - pd.Timedelta(minutes=30)
                # 24 hour window
                t_minus_24h = ts - pd.Timedelta(hours=24)

                v5 = sum(1 for t in past_ts if t >= t_minus_5m)
                v30_idx = [idx for idx, t in enumerate(past_ts) if t >= t_minus_30m]
                v30_count = len(v30_idx)
                v30_amt = sum(past_amts[idx] for idx in v30_idx)
                v24_count = sum(1 for t in past_ts if t >= t_minus_24h)

                transactions_last_5m[i] = v5
                transactions_last_30m[i] = v30_count
                amount_last_30m[i] = round(v30_amt, 2)
                transactions_last_24h[i] = v24_count

                # Novelty checks
                if dev and dev != "":
                    new_device[i] = 1 if dev not in past_devs else 0
                if merch and merch != "":
                    new_merchant[i] = 1 if merch not in past_merchs else 0
                if ip and ip != "":
                    new_ip[i] = 1 if ip not in past_ips else 0

            else:
                # First time seeing this customer
                customer_avg_amount[i] = amt
                customer_median_amount[i] = amt
                amount_vs_customer_avg[i] = 1.0
                amount_vs_customer_median[i] = 1.0
                customer_amount_zscore[i] = 0.0
                customer_transaction_count[i] = 0
                unique_device_count[i] = 0
                unique_merchant_count[i] = 0
                hour_deviation_from_customer_norm[i] = 0.0
                time_since_previous_txn[i] = -1.0
                transactions_last_5m[i] = 0
                transactions_last_30m[i] = 0
                amount_last_30m[i] = 0.0
                transactions_last_24h[i] = 0

                new_device[i] = 1 if dev else 0
                new_merchant[i] = 1 if merch else 0
                new_ip[i] = 1 if ip else 0

                # Initialize sets
                customer_amounts[cust] = []
                customer_timestamps[cust] = []
                customer_devices[cust] = set()
                customer_merchants[cust] = set()
                customer_ips[cust] = set()
                customer_hours[cust] = []

            # 2. Network entity linkages
            if dev and dev != "":
                if dev not in device_to_customers:
                    device_to_customers[dev] = set()
                device_to_customers[dev].add(cust)
                device_customer_count[i] = len(device_to_customers[dev])

            if ip and ip != "":
                if ip not in ip_to_customers:
                    ip_to_customers[ip] = set()
                ip_to_customers[ip].add(cust)
                ip_customer_count[i] = len(ip_to_customers[ip])

            # Entity degree estimate (connected unique entities)
            deg = 1  # itself
            if dev and dev != "":
                deg += 1
            if ip and ip != "":
                deg += 1
            if merch and merch != "":
                deg += 1
            entity_degree[i] = deg

            # Multi-customer sharing signals
            if device_customer_count[i] > 2 or ip_customer_count[i] > 3:
                suspicious_neighbor_count[i] += (device_customer_count[i] - 1) + (ip_customer_count[i] - 1)

            # 3. Append current transaction into customer history for future steps
            customer_amounts[cust].append(amt)
            customer_timestamps[cust].append(ts)
            customer_hours[cust].append(hr)
            if dev and dev != "":
                customer_devices[cust].add(dev)
            if merch and merch != "":
                customer_merchants[cust].add(merch)
            if ip and ip != "":
                customer_ips[cust].add(ip)

        # Assign computed series to dataframe
        data["customer_avg_amount"] = customer_avg_amount
        data["customer_median_amount"] = customer_median_amount
        data["amount_vs_customer_avg"] = amount_vs_customer_avg
        data["amount_vs_customer_median"] = amount_vs_customer_median
        data["customer_amount_zscore"] = customer_amount_zscore
        data["customer_transaction_count"] = customer_transaction_count
        data["unique_device_count"] = unique_device_count
        data["unique_merchant_count"] = unique_merchant_count
        data["hour_deviation_from_customer_norm"] = hour_deviation_from_customer_norm

        data["transactions_last_5m"] = transactions_last_5m
        data["transactions_last_30m"] = transactions_last_30m
        data["amount_last_30m"] = amount_last_30m
        data["transactions_last_24h"] = transactions_last_24h
        data["time_since_previous_txn"] = time_since_previous_txn

        data["new_device"] = new_device
        data["new_merchant"] = new_merchant
        data["new_ip"] = new_ip

        data["device_customer_count"] = device_customer_count
        data["ip_customer_count"] = ip_customer_count
        data["entity_degree"] = entity_degree
        data["suspicious_neighbor_count"] = suspicious_neighbor_count
        data["connected_case_count"] = connected_case_count

        # Build clean feature matrix for models
        feature_matrix = data[MODEL_FEATURES].copy().fillna(0.0)

        return data, feature_matrix
