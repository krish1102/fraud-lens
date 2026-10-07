"""
CSV Loader with robust delimiter, encoding, and header normalization.
"""

import io
import csv
from typing import Union, Tuple, Optional
import pandas as pd


class CSVLoader:
    """Safely loads CSV data from file path, bytes, or string."""

    @staticmethod
    def load_csv(
        source: Union[str, bytes, io.BytesIO, io.StringIO],
        encoding: Optional[str] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Loads CSV into a pandas DataFrame and returns metadata.
        """
        encodings_to_try = [encoding] if encoding else ["utf-8", "utf-8-sig", "latin1", "cp1252"]
        df = None
        last_error = None
        used_encoding = "utf-8"

        # If source is file path
        if isinstance(source, str) and not source.startswith(("{", "[", "transaction_id", "txn_id")):
            import os
            if os.path.exists(source):
                for enc in encodings_to_try:
                    try:
                        df = pd.read_csv(source, encoding=enc, low_memory=False)
                        used_encoding = enc
                        break
                    except UnicodeDecodeError as e:
                        last_error = e
                    except Exception as e:
                        last_error = e
                        break
            else:
                # String content
                source_io = io.StringIO(source)
                df = pd.read_csv(source_io, low_memory=False)
        elif isinstance(source, bytes):
            for enc in encodings_to_try:
                try:
                    source_str = source.decode(enc)
                    df = pd.read_csv(io.StringIO(source_str), low_memory=False)
                    used_encoding = enc
                    break
                except UnicodeDecodeError as e:
                    last_error = e
        elif isinstance(source, io.BytesIO):
            raw_bytes = source.getvalue()
            for enc in encodings_to_try:
                try:
                    source_str = raw_bytes.decode(enc)
                    df = pd.read_csv(io.StringIO(source_str), low_memory=False)
                    used_encoding = enc
                    break
                except UnicodeDecodeError as e:
                    last_error = e
        elif isinstance(source, io.StringIO):
            df = pd.read_csv(source, low_memory=False)

        if df is None:
            raise ValueError(f"Failed to load CSV source: {last_error}")

        # Clean column names (strip whitespace)
        df.columns = [str(c).strip() for c in df.columns]

        metadata = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "encoding": used_encoding
        }
        return df, metadata
