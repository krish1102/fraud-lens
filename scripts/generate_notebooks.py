"""
Generates the 7 educational, reproducible Jupyter Notebooks matching the Project Specification:
01_eda.ipynb
02_customer_layer.ipynb
03_feature_engineering.ipynb
04_fraud_model.ipynb
05_anomaly_detection.ipynb
06_network_analysis.ipynb
07_shap_evaluation.ipynb
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


# 1. 01_eda.ipynb
nb1_cells = [
    md_cell("# FraudLens — 01: Exploratory Data Analysis (EDA)\n## IEEE-CIS Benchmark Dataset & Online Transaction Ingestion\nThis notebook demonstrates exploratory analysis, missing value distributions, class imbalance, and transaction amounts."),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load development benchmark data
data_path = Path('../data/raw/development_transactions.csv')
if not data_path.exists():
    data_path = Path('../data/demo/sample_ecommerce_transactions.csv')
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")
df.head()"""),
    code_cell("""# Inspect class distribution and imbalance
if 'isFraud' in df.columns:
    print("Class distribution:")
    print(df['isFraud'].value_counts(normalize=True))
    
# Summary statistics of amount
df['amount'].describe()"""),
    code_cell("""# Visualizing amount distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['amount'], bins=40, kde=True, color='skyblue')
plt.title('Transaction Amount Distribution')

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['amount']), bins=40, kde=True, color='salmon')
plt.title('Log(1 + Amount) Distribution')
plt.tight_layout()
plt.show()""")
]

# 2. 02_customer_layer.ipynb
nb2_cells = [
    md_cell("# FraudLens — 02: Synthetic Analytical Customer Layer\n## Documenting IEEE-CIS Realities & Entity Groupings\nIEEE-CIS does not provide real customer identifiers. Here we document and demonstrate the synthetic customer layer construction."),
    code_cell("""import pandas as pd
from pathlib import Path

df = pd.read_csv('../data/raw/development_transactions.csv')
print(f"Unique synthetic customer IDs: {df['customer_id'].nunique()}")
print(f"Unique device IDs: {df['device_id'].nunique()}")
print(f"Unique IP addresses: {df['ip_address'].nunique()}")"""),
    code_cell("""# Analyze customer transaction frequency distribution
txns_per_cust = df['customer_id'].value_counts()
print(f"Median transactions per customer: {txns_per_cust.median()}")
print(f"Max transactions per customer: {txns_per_cust.max()}")
txns_per_cust.head(10)""")
]

# 3. 03_feature_engineering.ipynb
nb3_cells = [
    md_cell("# FraudLens — 03: Strictly Time-Aware Feature Engineering\n## Preventing Temporal Leakage in Rolling Baselines & Velocity\nThis notebook shows how `TimeAwareFeatureEngine` sorts by timestamp and prevents future leakage."),
    code_cell("""import sys
sys.path.append('..')
import pandas as pd
from src.features.time_aware_engine import TimeAwareFeatureEngine
from src.configs.settings import MODEL_FEATURES

df = pd.read_csv('../data/raw/development_transactions.csv')
engine = TimeAwareFeatureEngine()
enriched_df, feature_matrix = engine.build_features(df)
print(f"Enriched shape: {enriched_df.shape}")
print(f"Feature matrix shape: {feature_matrix.shape}")
feature_matrix.head()"""),
    code_cell("""# Inspect rolling velocity features
enriched_df[['customer_id', 'timestamp', 'amount', 'customer_avg_amount', 'transactions_last_30m', 'new_device']].head(15)""")
]

# 4. 04_fraud_model.ipynb
nb4_cells = [
    md_cell("# FraudLens — 04: Supervised Fraud Classification\n## LightGBM with Imbalance Handling, PR-AUC & Top-k Evaluation"),
    code_cell("""import sys
sys.path.append('..')
import pandas as pd
from src.models.supervised_model import SupervisedFraudModel
from src.features.time_aware_engine import TimeAwareFeatureEngine

df = pd.read_csv('../data/raw/development_transactions.csv')
engine = TimeAwareFeatureEngine()
enriched, X = engine.build_features(df)
y = df['isFraud']

# Split 80/20 time-respecting
split_idx = int(len(X) * 0.8)
X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]

model = SupervisedFraudModel()
metrics = model.train(X_train, y_train, X_val, y_val)
print("Supervised Model Evaluation Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")"""),
    code_cell("""# Predict probabilities on validation set
probs = model.predict_proba(X_val)
print(f"Validation sample probabilities: {probs[:10]}")""")
]

# 5. 05_anomaly_detection.ipynb
nb5_cells = [
    md_cell("# FraudLens — 05: Unsupervised Anomaly Detection\n## Isolation Forest Scoring on Behavioral Outliers"),
    code_cell("""import sys
sys.path.append('..')
import pandas as pd
from src.models.anomaly_model import UnsupervisedAnomalyModel
from src.features.time_aware_engine import TimeAwareFeatureEngine

df = pd.read_csv('../data/raw/development_transactions.csv')
engine = TimeAwareFeatureEngine()
_, X = engine.build_features(df)

anom_model = UnsupervisedAnomalyModel()
anom_model.fit(X)

scores = anom_model.score(X)
print(f"Anomaly score min: {scores.min()}, max: {scores.max()}, mean: {scores.mean():.4f}")"""),
    code_cell("""# Inspect top anomalies
top_anom_idx = scores.argsort()[-5:][::-1]
df.iloc[top_anom_idx][['transaction_id', 'customer_id', 'amount', 'device_id']]""")
]

# 6. 06_network_analysis.ipynb
nb6_cells = [
    md_cell("# FraudLens — 06: Heterogeneous Entity Graph & Network Analysis\n## Uncovering Shared Devices, IP Hubs, and Syndicate Clusters"),
    code_cell("""import sys
sys.path.append('..')
import pandas as pd
from src.graph.graph_engine import GraphEngine

df = pd.read_csv('../data/raw/development_transactions.csv')
graph = GraphEngine()
G = graph.build_from_dataframe(df)

print(f"Graph nodes: {G.number_of_nodes()}")
print(f"Graph edges: {G.number_of_edges()}")"""),
    code_cell("""# Inspect metrics for customer with shared device
sample_cust = df['customer_id'].iloc[10]
metrics = graph.get_customer_network_metrics(sample_cust)
print(f"Metrics for {sample_cust}:")
for k, v in metrics.items():
    print(f"  {k}: {v}")""")
]

# 7. 07_shap_evaluation.ipynb
nb7_cells = [
    md_cell("# FraudLens — 07: SHAP Local Explainability & Case Summaries\n## Tree SHAP Local Attributions and Human-Readable Narratives"),
    code_cell("""import sys
sys.path.append('..')
import pandas as pd
from src.models.supervised_model import SupervisedFraudModel
from src.explainability.shap_explainer import ShapExplainer
from src.features.time_aware_engine import TimeAwareFeatureEngine

model = SupervisedFraudModel()
model.load()

df = pd.read_csv('../data/raw/development_transactions.csv')
engine = TimeAwareFeatureEngine()
enriched, X = engine.build_features(df)

explainer = ShapExplainer(model.model)
row = X.iloc[0]
explanation = explainer.explain_instance(row, fraud_probability=0.88)

print("Generated Human-Readable Explanation:")
print(explanation['summary_text'])""")
]

all_notebooks = {
    "01_eda.ipynb": nb1_cells,
    "02_customer_layer.ipynb": nb2_cells,
    "03_feature_engineering.ipynb": nb3_cells,
    "04_fraud_model.ipynb": nb4_cells,
    "05_anomaly_detection.ipynb": nb5_cells,
    "06_network_analysis.ipynb": nb6_cells,
    "07_shap_evaluation.ipynb": nb7_cells
}

for name, cells in all_notebooks.items():
    nb_json = make_notebook(cells)
    with open(NOTEBOOKS_DIR / name, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=2)
    print(f"[+] Generated notebook: {NOTEBOOKS_DIR / name}")

print("All 7 notebooks generated successfully!")
