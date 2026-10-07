"""
Schema Analyzer: Profiles columns, data types, null rates, cardinality, and potential anomalies.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class SchemaAnalyzer:
    """Analyzes raw uploaded datasets to assist automated mapping and quality evaluation."""

    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Inspects DataFrame and returns detailed schema profile.
        """
        total_rows = len(df)
        columns_profile = {}
        suspicious_leakage_columns = []
        constant_columns = []

        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            null_pct = round(float(null_count / total_rows), 4) if total_rows > 0 else 0.0
            n_unique = int(series.nunique(dropna=True))

            # Detect potential target leakage
            lower_col = col.lower()
            if lower_col in ["isfraud", "is_fraud", "fraud_label", "target", "label", "fraud"]:
                suspicious_leakage_columns.append(col)

            # Detect constant columns
            if n_unique <= 1 and total_rows > 1:
                constant_columns.append(col)

            # Sample values (up to 3 non-null)
            valid_samples = series.dropna().head(3).tolist()
            # Convert timestamp/numpy objects to string for JSON serialization
            serializable_samples = [str(s) for s in valid_samples]

            # Inferred type
            dtype_str = str(series.dtype)
            inferred_type = "string"
            if pd.api.types.is_numeric_dtype(series):
                inferred_type = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(series):
                inferred_type = "datetime"
            else:
                # Try simple date probe
                if null_count < total_rows:
                    sample_val = str(valid_samples[0]) if valid_samples else ""
                    if any(char in sample_val for char in ["-", "/", ":"]) and len(sample_val) >= 8:
                        try:
                            pd.to_datetime(series.dropna().head(5))
                            inferred_type = "datetime"
                        except Exception:
                            inferred_type = "string"

            columns_profile[col] = {
                "name": col,
                "inferred_type": inferred_type,
                "raw_dtype": dtype_str,
                "null_count": null_count,
                "null_percentage": null_pct,
                "unique_count": n_unique,
                "sample_values": serializable_samples,
                "is_constant": bool(n_unique <= 1 and total_rows > 1)
            }

        return {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "columns": columns_profile,
            "leakage_candidates": suspicious_leakage_columns,
            "constant_columns": constant_columns
        }
