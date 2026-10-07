# FraudLens 🛡️

**Customer-Centric Fraud & Anomaly Detection with Explainable AI, Heterogeneous Entity Networks, and RAG-Powered Investigation**

---

## Overview

FraudLens is a production-grade fraud investigation intelligence platform built on three analytical pillars:

| Pillar | Technology | Purpose |
|---|---|---|
| **Supervised Detection** | LightGBM + class imbalance weighting | Probabilistic fraud pattern matching (PR-AUC: 0.98, ROC-AUC: 0.999) |
| **Unsupervised Anomaly** | Isolation Forest | Behavioral baseline deviation scoring |
| **Network Risk** | NetworkX heterogeneous graph | Shared device/IP/mule hub detection |

These three scores are combined via configurable weights into a **Combined Investigation Priority Score**, which drives automatic case generation and triage routing.

---

## Architecture

```
fraud-lens/
├── api/                     # FastAPI REST backend
│   └── main.py              # 14 endpoints (upload → score → cases → chat → export)
├── app/
│   └── streamlit_app.py     # 8-screen Streamlit analyst dashboard
├── src/
│   ├── configs/settings.py  # Canonical schema, feature definitions, risk weights
│   ├── ingestion/           # CSV Loader → Schema Analyzer → Column Mapper → Validator
│   ├── features/            # TimeAwareFeatureEngine (no future leakage)
│   ├── models/              # SupervisedFraudModel + UnsupervisedAnomalyModel + RiskEngine
│   ├── graph/               # GraphEngine – heterogeneous entity network
│   ├── explainability/      # ShapExplainer – Tree SHAP local attributions
│   ├── rag/                 # KnowledgeIndexer + RagRetriever + InvestigationCopilot
│   ├── storage/             # SQLite DatabaseManager (9 tables)
│   └── reporting/           # CaseExporter (Markdown + HTML)
├── knowledge_base/
│   ├── fraud_typologies/    # ATO, Card Testing, Velocity Anomaly, Mule, Device Spoofing
│   ├── investigation/       # Analyst SOP, Verification Guidelines, Escalation Matrix
│   ├── regulatory/          # RBI Master Directions 2024, EWS Guidance, FAQs
│   └── historical_cases/    # Synthetic benchmark cases (no real PII)
├── notebooks/               # 7 reproducible Jupyter notebooks
├── scripts/
│   ├── setup_dev_data_and_models.py   # One-shot initialization
│   └── generate_notebooks.py
├── tests/                   # 15 pytest tests (100% pass)
│   ├── test_api.py          # FastAPI endpoint integration tests
│   ├── test_ingestion.py    # CSV loader, alias mapper, validator
│   ├── test_leakage_and_features.py   # Strict leakage verification
│   ├── test_models.py       # Supervised, anomaly, risk engine, SHAP
│   └── test_graph_and_rag.py          # Graph and RAG copilot
├── models/                  # Persisted model artifacts (gitignored)
├── data/                    # Data files (gitignored)
└── .venv/                   # Virtual environment (gitignored)
```

---

## Quick Start

### 1. Create Virtual Environment & Install

```powershell
uv venv .venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install fastapi "uvicorn[standard]" streamlit xgboost lightgbm scikit-learn shap networkx pandas numpy pyarrow pydantic python-multipart httpx pytest plotly
```

### 2. Initialize Models & Demo Data (one-time)

```powershell
.venv\Scripts\python.exe scripts/setup_dev_data_and_models.py
```

This trains and saves:
- `models/supervised_fraud_model.joblib` — LightGBM fraud classifier (PR-AUC: **0.9818**)
- `models/anomaly_isolation_forest.joblib` — Isolation Forest anomaly scorer
- `data/rag_index.json` — 56 indexed knowledge chunks
- `data/demo/sample_ecommerce_transactions.csv` — 600-row demo dataset
- `data/demo/sample_unmapped_aliases.csv` — Column alias mapping demo dataset

### 3. Run the Streamlit Analyst Dashboard

```powershell
.venv\Scripts\streamlit run app/streamlit_app.py
```

### 4. Run the FastAPI Backend (optional)

```powershell
.venv\Scripts\uvicorn api.main:app --reload --port 8000
```

Swagger docs at: http://localhost:8000/docs

### 5. Run the Test Suite

```powershell
.venv\Scripts\pytest -v
```

Expected: **15 passed** in ~25 seconds.

---

## Key Features

### 🔍 Smart Schema Ingestion
- Auto-detects 10+ column alias families (`txn_id`, `cust_no`, `txn_amount`, `device`, `ip`...)
- Fuzzy substring matching for non-standard headers
- Canonical validation with quality reports and capability tier detection

### ⏱️ Time-Aware Feature Engineering (No Leakage)
- Customer baseline (mean, median, z-score) strictly computed from **prior** transactions only
- Rolling velocity: 5m, 30m, 24h windows computed without peeking forward
- Novelty flags: `new_device`, `new_merchant`, `new_ip` per customer history
- All 29 features reproducible with version freeze (`features_v1.1`)

### 🤖 Multi-Modal Risk Scoring
```
Combined Score = 0.50 × fraud_probability
               + 0.30 × anomaly_score  
               + 0.20 × network_risk
```
- **HIGH** (≥ 0.70) → Automatic case creation
- **MEDIUM** (0.40–0.70) → Queued for analyst review
- **LOW** (< 0.40) → Logged, frictionless pass

### 📈 SHAP Local Explanations
- Tree SHAP attributions identifying top positive/negative feature contributors
- Human-readable narrative: *"Amount 5.3x higher than customer average"*
- Explicit disclaimer: *"These signals do not by themselves constitute proof of fraudulent intent"*

### 🌐 Heterogeneous Entity Graph
- Bipartite graph: Customers ↔ Devices, IPs, Merchants, Beneficiaries
- Network risk heuristic incorporating shared device count, shared IP count, suspicious neighbor proximity
- Subgraph extraction (up to 2-hop ego networks) for visualization

### 🤝 RAG Investigation Copilot
- Indexes 56 structured knowledge chunks from 10 official documents
- TF-IDF + keyword boost hybrid retrieval
- Intent detection: `why_flagged`, `fraud_typology`, `investigation_steps`, `regulatory_guidance`, `similar_cases`
- All answers include full source citations with document IDs, source URLs, and publication dates

### 📚 Knowledge Base
| Category | Documents |
|---|---|
| Fraud Typologies | Account Takeover, Card Testing, Velocity Anomaly, Mule Account, Device Spoofing |
| Investigation SOPs | Analyst SOP, Verification Decision Matrix, Escalation & Law Enforcement Matrix |
| Regulatory | RBI Master Directions 2024, EWS Guidance, RBI FAQs |
| Historical Cases | 3 synthetic benchmark cases (ATO, Card Testing, Syndicate ring) |

### 💾 SQLite Persistence (9 Tables)
`datasets` · `column_mappings` · `transactions` · `customer_profiles` · `predictions` · `cases` · `case_notes` · `rag_documents` · `rag_interactions`

---

## FastAPI Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | System info & status |
| `GET` | `/stats` | Model metrics & case overview |
| `POST` | `/upload` | Upload raw CSV |
| `POST` | `/schema/analyze?dataset_id=` | Profile schema + suggest mappings |
| `POST` | `/schema/confirm` | Confirm canonical field mapping |
| `POST` | `/validate?dataset_id=` | Validate data quality & detect capabilities |
| `POST` | `/features/build?dataset_id=` | Run time-aware feature engineering |
| `POST` | `/predict?dataset_id=` | Score all transactions, create cases |
| `GET` | `/cases` | List all investigation cases |
| `GET` | `/case/{case_id}` | Case detail with SHAP + network evidence |
| `POST` | `/case/{case_id}/notes` | Update analyst notes & status |
| `POST` | `/case/{case_id}/chat` | RAG copilot investigation query |
| `GET` | `/network/{customer_id}` | Entity network metrics & ego graph |
| `GET` | `/export/{case_id}` | Export report (markdown/html/json) |

---

## Notebooks

| Notebook | Contents |
|---|---|
| `01_eda.ipynb` | EDA, class imbalance, amount distributions |
| `02_customer_layer.ipynb` | Synthetic customer layer construction |
| `03_feature_engineering.ipynb` | Time-aware feature engineering walkthrough |
| `04_fraud_model.ipynb` | LightGBM training, PR-AUC evaluation |
| `05_anomaly_detection.ipynb` | Isolation Forest scoring |
| `06_network_analysis.ipynb` | Entity graph construction & metrics |
| `07_shap_evaluation.ipynb` | SHAP attributions & case narratives |

---

## Model Performance

| Metric | Value |
|---|---|
| PR-AUC | **0.9818** |
| ROC-AUC | **0.9989** |
| Precision@Top 5% | 0.6667 |
| Recall@Top 5% | 0.9524 |
| F1 Score (thresh=0.5) | 0.9500 |
| Brier Score | 0.0033 |
| Anomaly Precision@Top 5% | **0.70** |

Trained on 2,400 synthetic development transactions; evaluated on 600-transaction hold-out.

---

## Regulatory Alignment (RBI)

FraudLens is designed to align with [RBI Master Directions on Fraud Risk Management (July 15, 2024)](https://systemhealth.rbi.org.in/Scripts/BS_ViewMasDirections.aspx):

- ✅ Automated EWS via multi-modal ML scoring
- ✅ Explainable AI alerts (SHAP narratives, not black-box)
- ✅ Entity linkage analytics (heterogeneous graph)
- ✅ Natural justice: automated models generate alerts, **humans make final fraud classification**
- ✅ Auditable case trail with analyst notes, RAG interaction log, timestamped decisions

---

## License

MIT License. See `LICENSE`.

> **Note**: All synthetic benchmark data in `data/` contains no real PII. Knowledge base documents are based on publicly available regulatory publications.
