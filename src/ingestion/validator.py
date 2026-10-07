"""
Data Validator: Enforces canonical schema constraints, validates types, nulls, duplicates, and detects analytics capabilities.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from src.configs.settings import (
    CANONICAL_FIELDS,
    MAX_NULL_RATE_REQUIRED,
    MAX_NULL_RATE_OPTIONAL
)


class DataValidator:
    """Validates mapped canonical DataFrame and determines analysis capability tier."""

    @staticmethod
    def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates the mapped canonical DataFrame, performs safe coercions,
        and returns cleaned DataFrame along with comprehensive validation report.
        """
        report: Dict[str, Any] = {
            "is_valid": True,
            "block_reasons": [],
            "warnings": [],
            "rows_initial": len(df),
            "rows_cleaned": len(df),
            "duplicate_transactions_count": 0,
            "failed_timestamp_rows": 0,
            "coerced_amount_rows": 0,
            "capabilities": {
                "basic": False,
                "behavioral": False,
                "network": False,
                "enabled_modules": []
            }
        }

        cleaned_df = df.copy()

        # 1. Required Columns Check
        missing_required = [
            f for f, spec in CANONICAL_FIELDS.items()
            if spec["required"] and f not in cleaned_df.columns
        ]
        if missing_required:
            report["is_valid"] = False
            report["block_reasons"].append(
                f"Missing required canonical column(s): {', '.join(missing_required)}"
            )
            return cleaned_df, report

        # 2. Check Empty Dataset
        if len(cleaned_df) == 0:
            report["is_valid"] = False
            report["block_reasons"].append("The uploaded dataset contains zero rows.")
            return cleaned_df, report

        # 3. String normalization for identifiers
        cleaned_df["transaction_id"] = cleaned_df["transaction_id"].astype(str).str.strip()
        cleaned_df["customer_id"] = cleaned_df["customer_id"].astype(str).str.strip()

        # 4. Duplicate Transaction ID Check
        dup_mask = cleaned_df["transaction_id"].duplicated(keep="first")
        dup_count = int(dup_mask.sum())
        if dup_count > 0:
            report["duplicate_transactions_count"] = dup_count
            report["warnings"].append(
                f"Found {dup_count} duplicate transaction_id values. Keeping first occurrence."
            )
            cleaned_df = cleaned_df[~dup_mask].copy()

        # 5. Amount Coercion and Plausibility
        raw_amounts = cleaned_df["amount"]
        cleaned_amounts = pd.to_numeric(raw_amounts, errors="coerce")
        coerced_fails = int(cleaned_amounts.isna().sum()) - int(raw_amounts.isna().sum())
        if coerced_fails > 0:
            report["coerced_amount_rows"] = coerced_fails
            report["warnings"].append(f"{coerced_fails} rows with non-numeric amount coerced to NaN.")

        # Check negative or zero amounts
        invalid_amounts = (cleaned_amounts <= 0).sum()
        if invalid_amounts > 0:
            report["warnings"].append(f"Found {int(invalid_amounts)} non-positive transaction amounts.")

        cleaned_df["amount"] = cleaned_amounts.fillna(0.0)

        # 6. Timestamp Parsing
        try:
            parsed_timestamps = pd.to_datetime(cleaned_df["timestamp"], errors="coerce")
            ts_fails = int(parsed_timestamps.isna().sum())
            if ts_fails > 0:
                report["failed_timestamp_rows"] = ts_fails
                report["warnings"].append(
                    f"{ts_fails} rows with invalid date/time format coerced or dropped."
                )
                # Drop unparseable timestamps if too few, else block
                if ts_fails / len(cleaned_df) > MAX_NULL_RATE_REQUIRED:
                    report["is_valid"] = False
                    report["block_reasons"].append(
                        f"Timestamp parsing failed for {ts_fails} rows ({round(ts_fails/len(cleaned_df)*100, 1)}% > threshold)."
                    )
                    return cleaned_df, report
            cleaned_df["timestamp"] = parsed_timestamps
            # Drop rows where timestamp could not be parsed
            cleaned_df = cleaned_df.dropna(subset=["timestamp"]).copy()
        except Exception as e:
            report["is_valid"] = False
            report["block_reasons"].append(f"Fatal error parsing timestamps: {str(e)}")
            return cleaned_df, report

        # 7. Null Rates Check
        for req in ["transaction_id", "customer_id", "timestamp", "amount"]:
            null_rate = cleaned_df[req].isna().mean()
            if null_rate > MAX_NULL_RATE_REQUIRED:
                report["is_valid"] = False
                report["block_reasons"].append(
                    f"Required field '{req}' has {round(null_rate*100, 1)}% missing values, exceeding limit."
                )

        # 8. Leakage check: ignore isFraud if present
        for leak_col in ["isFraud", "isfraud", "is_fraud"]:
            if leak_col in cleaned_df.columns:
                report["warnings"].append(
                    f"Column '{leak_col}' detected. Treated as ground-truth evaluation target, excluded from inference feature set."
                )

        # 9. Capability Detection (Basic, Behavioral, Network)
        cols = set(cleaned_df.columns)
        has_basic = {"transaction_id", "customer_id", "timestamp", "amount"}.issubset(cols)
        has_behavioral = has_basic and (("device_id" in cols and cleaned_df["device_id"].notna().sum() > 0) or
                                         ("merchant_id" in cols and cleaned_df["merchant_id"].notna().sum() > 0))
        has_network = has_behavioral and (
            ("device_id" in cols and cleaned_df["device_id"].nunique() > 1) or
            ("ip_address" in cols and cleaned_df["ip_address"].nunique() > 1)
        )

        report["capabilities"]["basic"] = bool(has_basic)
        report["capabilities"]["behavioral"] = bool(has_behavioral)
        report["capabilities"]["network"] = bool(has_network)

        enabled = ["amount_deviation", "time_anomaly", "customer_history"]
        if has_behavioral:
            enabled.extend(["velocity_analysis", "new_device_novelty", "merchant_novelty", "behavioral_baselines"])
        if has_network:
            enabled.extend(["entity_graph", "shared_device_clusters", "shared_ip_analysis", "graph_risk_features"])

        report["capabilities"]["enabled_modules"] = enabled
        report["rows_cleaned"] = len(cleaned_df)

        return cleaned_df, report
