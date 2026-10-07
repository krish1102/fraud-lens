"""
SQLite Storage Engine:
Implements persistent tables for datasets, column mappings, transactions, customer profiles,
model predictions, investigation cases, analyst notes, and RAG interactions.
"""

from typing import Dict, List, Any, Optional
import sqlite3
import json
from pathlib import Path
import pandas as pd
from src.configs.settings import DB_PATH


class DatabaseManager:
    """Manages SQLite tables and relational case queries."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes tables adhering to Technical Specification Section 17."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. datasets
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                upload_time TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                row_count INTEGER NOT NULL
            );
            """)

            # 2. column_mappings
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS column_mappings (
                mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                source_column TEXT NOT NULL,
                canonical_field TEXT NOT NULL,
                confidence REAL NOT NULL,
                confirmed_by_user INTEGER DEFAULT 1,
                FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
            );
            """)

            # 3. transactions
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                amount REAL NOT NULL,
                device_id TEXT,
                merchant_id TEXT,
                ip_address TEXT,
                location TEXT,
                beneficiary_id TEXT,
                payment_method TEXT,
                raw_json TEXT
            );
            """)

            # 4. customer_profiles
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_profiles (
                customer_id TEXT PRIMARY KEY,
                first_seen TEXT,
                last_seen TEXT,
                transaction_count INTEGER,
                avg_amount REAL,
                median_amount REAL,
                baseline_features_json TEXT
            );
            """)

            # 5. predictions
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                transaction_id TEXT PRIMARY KEY,
                model_version TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                anomaly_score REAL NOT NULL,
                network_risk REAL NOT NULL,
                combined_score REAL NOT NULL,
                risk_band TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
            );
            """)

            # 6. cases
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                anomaly_score REAL NOT NULL,
                network_risk REAL NOT NULL,
                combined_score REAL NOT NULL,
                risk_band TEXT NOT NULL,
                detected_signals TEXT,
                status TEXT NOT NULL DEFAULT 'Open',
                analyst_notes TEXT,
                model_version TEXT NOT NULL,
                rag_sources TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
            );
            """)

            # 7. case_notes
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS case_notes (
                note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                author TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                note TEXT NOT NULL,
                FOREIGN KEY (case_id) REFERENCES cases(case_id)
            );
            """)

            # 8. rag_documents
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                document_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT,
                jurisdiction TEXT,
                version TEXT,
                ingestion_date TEXT
            );
            """)

            # 9. rag_interactions
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                source_ids TEXT,
                timestamp TEXT NOT NULL
            );
            """)

            conn.commit()

    def register_dataset(self, dataset_id: str, filename: str, row_count: int, schema_version: str = "v1.0"):
        with self.get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO datasets VALUES (?, ?, datetime('now'), ?, ?)",
                (dataset_id, filename, schema_version, row_count)
            )
            conn.commit()

    def save_predictions_and_cases(self, scored_records: List[Dict[str, Any]]):
        """Batch inserts scored predictions and opens cases for High/Medium risk transactions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for r in scored_records:
                tid = r["transaction_id"]
                cid = r["customer_id"]
                mod_ver = r.get("model_version", "v1.2")
                p_fraud = float(r.get("fraud_probability", 0.0))
                s_anom = float(r.get("anomaly_score", 0.0))
                s_net = float(r.get("network_risk", 0.0))
                c_score = float(r.get("combined_score", 0.0))
                r_band = r.get("risk_band", "LOW")
                signals_json = json.dumps(r.get("detected_signals", []))

                # Insert prediction
                cursor.execute("""
                INSERT OR REPLACE INTO predictions 
                (transaction_id, model_version, fraud_probability, anomaly_score, network_risk, combined_score, risk_band, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """, (tid, mod_ver, p_fraud, s_anom, s_net, c_score, r_band))

                # If High or Medium risk, create or update case
                if r_band in ["HIGH", "MEDIUM"]:
                    case_id = f"FRD-{tid.lstrip('T').zfill(6)}"
                    cursor.execute("""
                    INSERT OR IGNORE INTO cases
                    (case_id, transaction_id, customer_id, fraud_probability, anomaly_score, network_risk, combined_score, risk_band, detected_signals, status, model_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Open', ?, datetime('now'))
                    """, (case_id, tid, cid, p_fraud, s_anom, s_net, c_score, r_band, signals_json, mod_ver))

            conn.commit()

    def get_all_cases(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cases ORDER BY combined_score DESC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            if res.get("detected_signals"):
                try:
                    res["detected_signals"] = json.loads(res["detected_signals"])
                except Exception:
                    pass
            return res

    def update_case_notes(self, case_id: str, notes: str, status: Optional[str] = None):
        with self.get_connection() as conn:
            if status:
                conn.execute("UPDATE cases SET analyst_notes = ?, status = ? WHERE case_id = ?", (notes, status, case_id))
            else:
                conn.execute("UPDATE cases SET analyst_notes = ? WHERE case_id = ?", (notes, case_id))
            conn.commit()

    def log_rag_interaction(self, case_id: str, question: str, answer: str, source_ids: List[str]):
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO rag_interactions (case_id, question, answer, source_ids, timestamp) VALUES (?, ?, ?, ?, datetime('now'))",
                (case_id, question, answer, json.dumps(source_ids))
            )
            conn.commit()
