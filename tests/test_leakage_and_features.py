"""
Leakage and Feature Engineering Tests:
Verifies that customer baseline, velocity, and novelty features strictly prevent temporal leakage.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.time_aware_engine import TimeAwareFeatureEngine
from src.configs.settings import MODEL_FEATURES
from src.models.training_pipeline import TrainingPipeline


def test_time_aware_baseline_no_future_leakage():
    """
    Ensures that for a customer with transactions at t0, t1, t2:
    - At t0: customer has 0 prior transactions, baseline is default
    - At t1: customer baseline uses ONLY t0 amount (ignores t2)
    - At t2: customer baseline uses mean(t0, t1)
    """
    engine = TimeAwareFeatureEngine()

    df = pd.DataFrame([
        {
            "transaction_id": "T0",
            "customer_id": "C_TEST",
            "timestamp": "2026-09-01 10:00:00",
            "amount": 100.0,
            "device_id": "D1",
            "merchant_id": "M1",
            "ip_address": "1.1.1.1"
        },
        {
            "transaction_id": "T1",
            "customer_id": "C_TEST",
            "timestamp": "2026-09-01 10:10:00",
            "amount": 50.0,
            "device_id": "D1",
            "merchant_id": "M1",
            "ip_address": "1.1.1.1"
        },
        {
            "transaction_id": "T2",
            "customer_id": "C_TEST",
            "timestamp": "2026-09-01 10:20:00",
            "amount": 1000.0,  # Huge future transaction!
            "device_id": "D2",  # New device!
            "merchant_id": "M2",
            "ip_address": "2.2.2.2"
        }
    ])

    enriched, feat_mat = engine.build_features(df)

    # 1. At T0 (first transaction)
    row_t0 = enriched[enriched["transaction_id"] == "T0"].iloc[0]
    assert row_t0["customer_transaction_count"] == 0
    assert row_t0["new_device"] == 1  # First time device seen
    assert row_t0["transactions_last_30m"] == 0

    # 2. At T1: baseline must reflect ONLY T0 ($100), not T2 ($1000)!
    row_t1 = enriched[enriched["transaction_id"] == "T1"].iloc[0]
    assert row_t1["customer_transaction_count"] == 1
    assert row_t1["customer_avg_amount"] == 100.0
    assert row_t1["amount_vs_customer_avg"] == 0.5  # 50 / 100 = 0.5
    assert row_t1["new_device"] == 0  # D1 was seen at T0
    assert row_t1["transactions_last_30m"] == 1  # T0 occurred 10 min ago

    # 3. At T2: baseline reflects mean(100, 50) = 75.0
    row_t2 = enriched[enriched["transaction_id"] == "T2"].iloc[0]
    assert row_t2["customer_transaction_count"] == 2
    assert row_t2["customer_avg_amount"] == 75.0
    assert row_t2["new_device"] == 1  # D2 is new for this customer!
    assert row_t2["transactions_last_30m"] == 2


def test_feature_matrix_reproducible_columns():
    engine = TimeAwareFeatureEngine()
    df = pd.DataFrame([
        {
            "transaction_id": "T1",
            "customer_id": "C1",
            "timestamp": "2026-09-01 12:00:00",
            "amount": 25.0
        }
    ])
    _, feat_mat = engine.build_features(df)
    assert list(feat_mat.columns) == list(MODEL_FEATURES)
    assert len(feat_mat) == 1


def test_synthetic_demo_contains_customer_history(tmp_path):
    pipeline = TrainingPipeline(models_dir=tmp_path)
    demo_df = pipeline.generate_synthetic_development_data(n_rows=600, random_seed=99)

    transactions_per_customer = demo_df["customer_id"].value_counts()
    assert len(demo_df) == 600
    assert demo_df["customer_id"].nunique() == 120
    assert transactions_per_customer.min() >= 3
    assert demo_df["isFraud"].nunique() == 2

    historical_df, _ = TimeAwareFeatureEngine().build_features(
        demo_df.drop(columns=["isFraud"])
    )
    assert historical_df["customer_transaction_count"].max() >= 2
    assert (historical_df["transactions_last_30m"] > 0).any()
