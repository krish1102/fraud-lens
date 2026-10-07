"""
Script to initialize demo datasets, train the base supervised and anomaly models, and build RAG index.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.settings import DATA_DIR
from src.models.training_pipeline import TrainingPipeline
from src.rag.knowledge_indexer import KnowledgeIndexer


def main():
    print("=" * 60)
    print("FRAUDLENS INITIALIZATION & ARTIFACT GENERATOR")
    print("=" * 60)

    demo_dir = DATA_DIR / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Canonical CSV Template
    template_path = demo_dir / "canonical_template.csv"
    template_df = pd.DataFrame([
        {
            "transaction_id": "T001928",
            "customer_id": "C00042",
            "timestamp": "2026-09-01 10:30:00",
            "amount": 1250.50,
            "merchant_id": "M0028",
            "device_id": "D091",
            "ip_address": "203.0.113.10",
            "location": "Delhi",
            "beneficiary_id": "B441",
            "payment_method": "card"
        },
        {
            "transaction_id": "T001929",
            "customer_id": "C00042",
            "timestamp": "2026-09-01 10:35:12",
            "amount": 45.00,
            "merchant_id": "M0015",
            "device_id": "D091",
            "ip_address": "203.0.113.10",
            "location": "Delhi",
            "beneficiary_id": "",
            "payment_method": "upi"
        }
    ])
    template_df.to_csv(template_path, index=False)
    print(f"[+] Created canonical template: {template_path}")

    # 2. Train and save Supervised & Anomaly models using TrainingPipeline
    print("[*] Running Training Pipeline on development benchmark data...")
    pipeline = TrainingPipeline()
    # Generate dev training data (3,000 transactions)
    dev_data = pipeline.generate_synthetic_development_data(n_rows=3000, random_seed=42)
    dev_data_path = DATA_DIR / "raw" / "development_transactions.csv"
    dev_data_path.parent.mkdir(parents=True, exist_ok=True)
    dev_data.to_csv(dev_data_path, index=False)
    print(f"[+] Saved development benchmark dataset: {dev_data_path}")

    manifest = pipeline.train_and_persist(dev_data)
    print(f"[+] Models trained successfully! PR-AUC: {manifest['supervised_metrics']['pr_auc']:.4f}, ROC-AUC: {manifest['supervised_metrics']['roc_auc']:.4f}")

    # 3. Create Sample E-commerce User Prediction CSV (without isFraud target)
    print("[*] Generating rich realistic demo user prediction dataset...")
    user_demo_df = pipeline.generate_synthetic_development_data(n_rows=600, random_seed=99)
    # Remove isFraud column for user prediction demonstration
    clean_user_df = user_demo_df.drop(columns=["isFraud"])
    sample_csv_path = demo_dir / "sample_ecommerce_transactions.csv"
    clean_user_df.to_csv(sample_csv_path, index=False)
    print(f"[+] Saved realistic user transaction dataset: {sample_csv_path}")

    # 4. Create Sample Unmapped Aliases CSV to demonstrate Schema Analyzer & Column Mapper
    aliases_df = pd.DataFrame({
        "trans_id": clean_user_df["transaction_id"].head(50),
        "cust_no": clean_user_df["customer_id"].head(50),
        "datetime": clean_user_df["timestamp"].head(50),
        "txn_value": clean_user_df["amount"].head(50),
        "device": clean_user_df["device_id"].head(50),
        "merchant": clean_user_df["merchant_id"].head(50),
        "ip": clean_user_df["ip_address"].head(50)
    })
    aliases_path = demo_dir / "sample_unmapped_aliases.csv"
    aliases_df.to_csv(aliases_path, index=False)
    print(f"[+] Saved unmapped aliases demo dataset: {aliases_path}")

    # 5. Build RAG Index
    print("[*] Building RAG knowledge base vector index...")
    indexer = KnowledgeIndexer()
    chunk_count = indexer.build_index()
    print(f"[+] Indexed {chunk_count} structured knowledge chunks into {indexer.index_path}")

    print("=" * 60)
    print("SETUP COMPLETE! All models and datasets ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
