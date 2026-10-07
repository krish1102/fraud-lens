"""
FraudLens Global Configuration and Schema Definitions
"""

from typing import Dict, List, Any
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
DB_PATH = BASE_DIR / "fraudlens.db"

# Canonical Field Definitions
CANONICAL_FIELDS = {
    "transaction_id": {
        "required": True,
        "type": "string",
        "description": "Unique identifier for the transaction record",
        "example": "T001928"
    },
    "customer_id": {
        "required": True,
        "type": "string",
        "description": "Synthetic or real analytical customer identifier",
        "example": "C00042"
    },
    "timestamp": {
        "required": True,
        "type": "datetime",
        "description": "Transaction date and time",
        "example": "2026-09-01 10:30:00"
    },
    "amount": {
        "required": True,
        "type": "numeric",
        "description": "Transaction monetary amount",
        "example": 1250.50
    },
    "merchant_id": {
        "required": False,
        "type": "string",
        "description": "Merchant or payee identifier",
        "example": "M0028"
    },
    "device_id": {
        "required": False,
        "type": "string",
        "description": "Hardware device fingerprint or cookie ID",
        "example": "D091"
    },
    "ip_address": {
        "required": False,
        "type": "string",
        "description": "Client IP address",
        "example": "203.0.113.10"
    },
    "location": {
        "required": False,
        "type": "string",
        "description": "Geographic city, state, or region",
        "example": "Delhi"
    },
    "beneficiary_id": {
        "required": False,
        "type": "string",
        "description": "Receiving account or beneficiary ID",
        "example": "B441"
    },
    "payment_method": {
        "required": False,
        "type": "string",
        "description": "Card, UPI, Wallet, NetBanking, etc.",
        "example": "card"
    }
}

# Column Aliases for Flexible CSV Ingestion
COLUMN_ALIASES: Dict[str, List[str]] = {
    "transaction_id": [
        "transaction_id", "txn_id", "transactionid", "trans_id", "tx_id", "id", "transaction_number"
    ],
    "customer_id": [
        "customer_id", "cust_id", "cust_no", "customer_number", "account_id", "user_id", "client_id"
    ],
    "amount": [
        "amount", "txn_amount", "txn_value", "transaction_value", "payment_amount", "val", "price", "amt"
    ],
    "timestamp": [
        "timestamp", "txn_time", "txn_datetime", "datetime", "date_time", "transaction_time", "time", "date"
    ],
    "device_id": [
        "device_id", "device", "deviceid", "device_fingerprint", "hardware_id", "fingerprint"
    ],
    "merchant_id": [
        "merchant_id", "merchant", "merchantid", "seller_id", "vendor_id", "store_id"
    ],
    "ip_address": [
        "ip_address", "ip", "ipaddress", "client_ip", "remote_addr"
    ],
    "location": [
        "location", "city", "geo", "billing_city", "country", "region"
    ],
    "beneficiary_id": [
        "beneficiary_id", "beneficiary", "payee_id", "target_account", "recipient_id"
    ],
    "payment_method": [
        "payment_method", "payment_type", "channel", "method", "card_type", "instrument"
    ]
}

# Engineered Feature Schema (Versioned & Frozen for Model Reproducibility)
FEATURE_VERSION = "features_v1.1"
MODEL_VERSION = "fraud_model_v1.2"

MODEL_FEATURES: List[str] = [
    # Transaction level
    "transaction_amount",
    "log_transaction_amount",
    "hour",
    "day_of_week",
    "is_night",
    "is_weekend",
    # Customer baseline (time-aware historical)
    "customer_avg_amount",
    "customer_median_amount",
    "amount_vs_customer_avg",
    "amount_vs_customer_median",
    "customer_amount_zscore",
    "customer_transaction_count",
    "unique_device_count",
    "unique_merchant_count",
    "hour_deviation_from_customer_norm",
    # Velocity (strictly rolling backward)
    "transactions_last_5m",
    "transactions_last_30m",
    "amount_last_30m",
    "transactions_last_24h",
    "time_since_previous_txn",
    # Novelty
    "new_device",
    "new_merchant",
    "new_ip",
    # Network features
    "device_customer_count",
    "ip_customer_count",
    "entity_degree",
    "suspicious_neighbor_count",
    "connected_case_count"
]

# Risk Scoring Configurable Weights
DEFAULT_RISK_WEIGHTS = {
    "fraud_probability": 0.50,
    "anomaly_score": 0.30,
    "network_risk": 0.20
}

# Risk Band Cutoffs
RISK_BANDS = {
    "HIGH": 0.70,
    "MEDIUM": 0.40,
    "LOW": 0.00
}

# Validation Thresholds
MAX_NULL_RATE_REQUIRED = 0.05  # 5% max missing for required columns
MAX_NULL_RATE_OPTIONAL = 0.70  # 70% max missing for optional
