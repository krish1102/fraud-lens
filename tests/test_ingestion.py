"""
Unit tests for data ingestion, schema analysis, alias mapping, and validation.
"""

import pytest
import pandas as pd
from src.ingestion.loader import CSVLoader
from src.ingestion.schema_analyzer import SchemaAnalyzer
from src.ingestion.column_mapper import ColumnMapper
from src.ingestion.validator import DataValidator


def test_column_mapper_exact_and_alias():
    mapper = ColumnMapper()
    raw_cols = ["txn_id", "cust_no", "datetime", "txn_amount", "device", "merchant", "ip"]
    res = mapper.suggest_mappings(raw_cols)

    assert res["mapped"]["transaction_id"] == "txn_id"
    assert res["mapped"]["customer_id"] == "cust_no"
    assert res["mapped"]["timestamp"] == "datetime"
    assert res["mapped"]["amount"] == "txn_amount"
    assert res["mapped"]["device_id"] == "device"
    assert res["mapped"]["merchant_id"] == "merchant"
    assert res["mapped"]["ip_address"] == "ip"
    assert len(res["missing_required"]) == 0


def test_schema_analyzer_profiling():
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3"],
        "customer_id": ["C1", "C2", "C3"],
        "timestamp": ["2026-09-01 10:00:00", "2026-09-01 10:05:00", "2026-09-01 10:10:00"],
        "amount": [10.5, 20.0, 30.2],
        "isFraud": [0, 0, 1]
    })
    profile = SchemaAnalyzer.analyze(df)

    assert profile["total_rows"] == 3
    assert profile["total_columns"] == 5
    assert "isFraud" in profile["leakage_candidates"]
    assert profile["columns"]["amount"]["inferred_type"] == "numeric"


def test_validator_detects_missing_required():
    bad_df = pd.DataFrame({
        "customer_id": ["C1"],
        "amount": [50.0]
    })
    _, report = DataValidator.validate_and_clean(bad_df)
    assert report["is_valid"] is False
    assert any("transaction_id" in r for r in report["block_reasons"])


def test_validator_deduplicates_and_coerces():
    dup_df = pd.DataFrame({
        "transaction_id": ["T1", "T1", "T2"],
        "customer_id": ["C1", "C1", "C2"],
        "timestamp": ["2026-09-01 10:00:00", "2026-09-01 10:00:00", "2026-09-01 10:05:00"],
        "amount": ["100.50", "100.50", "invalid_amt"]
    })
    cleaned, report = DataValidator.validate_and_clean(dup_df)
    assert report["is_valid"] is True
    assert report["duplicate_transactions_count"] == 1
    assert len(cleaned) == 2
    assert report["coerced_amount_rows"] == 1
