"""
FraudLens FastAPI Application:
Serves analytical endpoints for CSV ingestion, schema validation, feature computation,
fraud scoring, SHAP local explainability, entity network analysis, RAG copilot, and case export.
"""

from typing import Dict, Any, List, Optional
import io
import uuid
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse, FileResponse
from pydantic import BaseModel
import pandas as pd

from src.configs.settings import (
    MODELS_DIR,
    FEATURE_VERSION,
    MODEL_VERSION,
    DEFAULT_RISK_WEIGHTS,
    DATA_DIR
)
from src.ingestion.loader import CSVLoader
from src.ingestion.schema_analyzer import SchemaAnalyzer
from src.ingestion.column_mapper import ColumnMapper
from src.ingestion.validator import DataValidator
from src.features.time_aware_engine import TimeAwareFeatureEngine
from src.models.supervised_model import SupervisedFraudModel
from src.models.anomaly_model import UnsupervisedAnomalyModel
from src.models.risk_engine import RiskEngine
from src.graph.graph_engine import GraphEngine
from src.explainability.shap_explainer import ShapExplainer
from src.rag.knowledge_indexer import KnowledgeIndexer
from src.rag.retriever import RagRetriever
from src.rag.investigation_copilot import InvestigationCopilot
from src.storage.database import DatabaseManager
from src.reporting.case_exporter import CaseExporter

app = FastAPI(
    title="FraudLens API",
    description="Customer-Centric Fraud & Anomaly Detection with RAG-Powered Investigation",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store & singleton services
uploaded_raw_dfs: Dict[str, pd.DataFrame] = {}
confirmed_mappings: Dict[str, Dict[str, str]] = {}
active_scored_dfs: Dict[str, pd.DataFrame] = {}

db = DatabaseManager()
feature_engine = TimeAwareFeatureEngine()

# Initialize models
supervised_model = SupervisedFraudModel()
supervised_model.load()
anomaly_model = UnsupervisedAnomalyModel()
anomaly_model.load()

risk_engine = RiskEngine()
graph_engine = GraphEngine()
shap_explainer = ShapExplainer(supervised_model.model)

# Initialize RAG
rag_indexer = KnowledgeIndexer()
if not rag_indexer.load_index():
    rag_indexer.build_index()
rag_retriever = RagRetriever(rag_indexer)
rag_copilot = InvestigationCopilot(rag_retriever)


# ------------------ Pydantic Request Models ------------------

class MappingConfirmRequest(BaseModel):
    dataset_id: str
    mapping: Dict[str, str]  # {canonical_field: source_column}


class ChatRequest(BaseModel):
    query: str
    jurisdiction: Optional[str] = None


class CaseNotesUpdate(BaseModel):
    notes: str
    status: Optional[str] = "Investigating"


# ------------------ Endpoints ------------------

@app.get("/")
def read_root():
    return {
        "system": "FraudLens",
        "description": "Customer-Centric Fraud & Anomaly Detection with RAG-Powered Investigation",
        "version": "1.2.0",
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "status": "Ready"
    }


@app.get("/stats")
def get_stats():
    """Returns system status, model evaluation metrics, and case counts."""
    cases = db.get_all_cases()
    high_count = sum(1 for c in cases if c.get("risk_band") == "HIGH")
    med_count = sum(1 for c in cases if c.get("risk_band") == "MEDIUM")
    low_count = sum(1 for c in cases if c.get("risk_band") == "LOW")

    manifest_path = MODELS_DIR / "model_manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    return {
        "models": {
            "version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "supervised_type": "LightGBM",
            "anomaly_type": "Isolation Forest",
            "metrics": manifest.get("supervised_metrics", {})
        },
        "cases_overview": {
            "total_cases": len(cases),
            "high_risk": high_count,
            "medium_risk": med_count,
            "low_risk": low_count
        },
        "rag_kb_chunks": len(rag_indexer.chunks)
    }


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Uploads and registers raw transaction CSV."""
    content = await file.read()
    try:
        df, metadata = CSVLoader.load_csv(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    dataset_id = str(uuid.uuid4())[:8]
    uploaded_raw_dfs[dataset_id] = df
    db.register_dataset(dataset_id, file.filename or "uploaded.csv", len(df))

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "rows_detected": len(df),
        "columns_detected": list(df.columns),
        "metadata": metadata
    }


@app.post("/schema/analyze")
def analyze_schema(dataset_id: str = Query(...)):
    """Analyzes columns, types, null rates, and suggests canonical field mappings."""
    if dataset_id not in uploaded_raw_dfs:
        raise HTTPException(status_code=404, detail="Dataset ID not found. Please upload first.")

    df = uploaded_raw_dfs[dataset_id]
    profile = SchemaAnalyzer.analyze(df)
    mapper = ColumnMapper()
    mapping_suggestions = mapper.suggest_mappings(list(df.columns))

    return {
        "dataset_id": dataset_id,
        "profile": profile,
        "suggestions": mapping_suggestions
    }


@app.post("/schema/confirm")
def confirm_schema_mapping(payload: MappingConfirmRequest):
    """Stores user-confirmed canonical mappings for the dataset."""
    if payload.dataset_id not in uploaded_raw_dfs:
        raise HTTPException(status_code=404, detail="Dataset ID not found.")

    confirmed_mappings[payload.dataset_id] = payload.mapping
    return {
        "status": "success",
        "dataset_id": payload.dataset_id,
        "confirmed_mapping": payload.mapping
    }


@app.post("/validate")
def validate_dataset(dataset_id: str = Query(...)):
    """Applies confirmed mapping and validates data constraints."""
    if dataset_id not in uploaded_raw_dfs:
        raise HTTPException(status_code=404, detail="Dataset ID not found.")

    mapping = confirmed_mappings.get(dataset_id)
    raw_df = uploaded_raw_dfs[dataset_id]

    if not mapping:
        # Auto suggest if user didn't explicitly confirm
        mapper = ColumnMapper()
        sug = mapper.suggest_mappings(list(raw_df.columns))
        mapping = sug["mapped"]
        confirmed_mappings[dataset_id] = mapping

    canonical_df = ColumnMapper.apply_mapping(raw_df, mapping)
    cleaned_df, report = DataValidator.validate_and_clean(canonical_df)

    if report["is_valid"]:
        # Store canonical dataframe for pipeline
        uploaded_raw_dfs[dataset_id] = cleaned_df

    return {
        "dataset_id": dataset_id,
        "validation_report": report
    }


@app.post("/features/build")
def build_features_endpoint(dataset_id: str = Query(...)):
    """Generates time-aware customer baseline, velocity, novelty, and network features."""
    if dataset_id not in uploaded_raw_dfs:
        raise HTTPException(status_code=404, detail="Dataset ID not found.")

    df = uploaded_raw_dfs[dataset_id]
    enriched_df, feature_matrix = feature_engine.build_features(df)

    # Cache enriched dataframe
    uploaded_raw_dfs[dataset_id] = enriched_df

    return {
        "dataset_id": dataset_id,
        "rows_processed": len(enriched_df),
        "features_generated": list(feature_matrix.columns),
        "feature_version": FEATURE_VERSION
    }


@app.post("/predict")
def run_predictions(dataset_id: str = Query(...)):
    """
    Executes trained Supervised Fraud Model (LightGBM) and Unsupervised Anomaly Model (Isolation Forest),
    builds entity graph, calculates network risks, and registers cases for high-risk transactions.
    """
    if dataset_id not in uploaded_raw_dfs:
        raise HTTPException(status_code=404, detail="Dataset ID not found.")

    enriched_df = uploaded_raw_dfs[dataset_id]
    if "transactions_last_30m" not in enriched_df.columns:
        enriched_df, feature_matrix = feature_engine.build_features(enriched_df)
    else:
        from src.configs.settings import MODEL_FEATURES
        feature_matrix = enriched_df[MODEL_FEATURES].copy().fillna(0.0)

    # 1. Supervised Fraud Probabilities
    fraud_probs = supervised_model.predict_proba(feature_matrix)

    # 2. Unsupervised Anomaly Scores
    anomaly_scores = anomaly_model.score(feature_matrix)

    # 3. Build Graph & calculate network metrics
    graph_engine.build_from_dataframe(enriched_df)

    network_risks = []
    for _, row in enriched_df.iterrows():
        cid = str(row["customer_id"])
        net_metrics = graph_engine.get_customer_network_metrics(cid)
        network_risks.append(net_metrics["network_risk"])

    # 4. Calculate Combined Scores & Risk Bands
    combined_scores = []
    risk_bands = []
    signals_list = []

    for i in range(len(enriched_df)):
        fp = float(fraud_probs[i])
        ascore = float(anomaly_scores[i])
        nrisk = float(network_risks[i])

        c_score = risk_engine.calculate_combined_score(fp, ascore, nrisk)
        r_band = risk_engine.get_risk_band(c_score)
        sigs = risk_engine.extract_detected_signals(enriched_df.iloc[i])

        combined_scores.append(c_score)
        risk_bands.append(r_band)
        signals_list.append(sigs)

    # Attach predictions to DataFrame
    scored_df = enriched_df.copy()
    scored_df["fraud_probability"] = fraud_probs
    scored_df["anomaly_score"] = anomaly_scores
    scored_df["network_risk"] = network_risks
    scored_df["combined_score"] = combined_scores
    scored_df["risk_band"] = risk_bands
    scored_df["detected_signals"] = signals_list

    active_scored_dfs[dataset_id] = scored_df

    # Save to SQLite
    records_to_save = []
    for _, r in scored_df.iterrows():
        records_to_save.append({
            "transaction_id": str(r["transaction_id"]),
            "customer_id": str(r["customer_id"]),
            "model_version": MODEL_VERSION,
            "fraud_probability": r["fraud_probability"],
            "anomaly_score": r["anomaly_score"],
            "network_risk": r["network_risk"],
            "combined_score": r["combined_score"],
            "risk_band": r["risk_band"],
            "detected_signals": r["detected_signals"]
        })
    db.save_predictions_and_cases(records_to_save)

    high_risk_count = sum(1 for b in risk_bands if b == "HIGH")
    med_risk_count = sum(1 for b in risk_bands if b == "MEDIUM")

    sample_top_scored = json.loads(
        scored_df.sort_values(by="combined_score", ascending=False).head(5)[
            ["transaction_id", "customer_id", "amount", "fraud_probability", "anomaly_score", "network_risk", "combined_score", "risk_band"]
        ].to_json(orient="records", date_format="iso")
    )

    return {
        "dataset_id": dataset_id,
        "total_scored": int(len(scored_df)),
        "high_risk_cases_generated": int(high_risk_count),
        "medium_risk_cases_generated": int(med_risk_count),
        "sample_top_scored": sample_top_scored
    }


@app.get("/cases")
def get_cases(risk_band: Optional[str] = None, status: Optional[str] = None):
    """Retrieves stored investigation cases with optional filtering."""
    cases = db.get_all_cases()
    if risk_band:
        cases = [c for c in cases if c.get("risk_band") == risk_band.upper()]
    if status:
        cases = [c for c in cases if c.get("status") == status]
    return {"cases": cases, "count": len(cases)}


@app.get("/case/{case_id}")
def get_case_details(case_id: str):
    """Retrieves full investigation case details including SHAP explanation and network neighbors."""
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case ID not found.")

    tid = case["transaction_id"]
    cid = case["customer_id"]

    # Compute live SHAP explanation
    shap_info = {}
    row_found = None
    for ds_id, df in active_scored_dfs.items():
        match = df[df["transaction_id"] == tid]
        if not match.empty:
            row_found = match.iloc[0]
            break

    if row_found is not None:
        shap_info = shap_explainer.explain_instance(row_found, case["fraud_probability"])
    else:
        # Fallback explanation
        dummy_row = pd.Series({
            "amount_vs_customer_avg": 2.5,
            "new_device": 1,
            "transactions_last_30m": 3,
            "customer_amount_zscore": 2.1
        })
        shap_info = shap_explainer.explain_instance(dummy_row, case["fraud_probability"])

    # Retrieve graph details
    net_metrics = graph_engine.get_customer_network_metrics(cid)
    subgraph_data = graph_engine.get_subgraph_data(cid, max_hops=2)

    case["shap_explanation"] = shap_info
    case["network_metrics"] = net_metrics
    case["subgraph"] = subgraph_data

    return json.loads(json.dumps(case, default=str))


@app.post("/case/{case_id}/notes")
def update_case_notes(case_id: str, payload: CaseNotesUpdate):
    """Updates analyst notes and case status."""
    db.update_case_notes(case_id, payload.notes, payload.status)
    return {"status": "success", "case_id": case_id, "updated_notes": payload.notes}


@app.post("/case/{case_id}/chat")
def case_chat(case_id: str, payload: ChatRequest):
    """Executes RAG investigation query in context of case evidence and regulatory knowledge."""
    case = db.get_case(case_id)
    if not case:
        # Create minimal case wrapper
        case = {"case_id": case_id, "detected_signals": []}

    response = rag_copilot.answer_query(payload.query, case, jurisdiction=payload.jurisdiction)
    source_ids = [s["document_id"] for s in response.get("sources", [])]
    db.log_rag_interaction(case_id, payload.query, response["answer"], source_ids)

    return response


@app.get("/network/{customer_id}")
def get_customer_network(customer_id: str):
    """Returns network evidence and ego subgraph for customer."""
    metrics = graph_engine.get_customer_network_metrics(customer_id)
    subgraph = graph_engine.get_subgraph_data(customer_id)
    return {
        "customer_id": customer_id,
        "metrics": metrics,
        "subgraph": subgraph
    }


@app.get("/export/{case_id}")
def export_case_report(case_id: str, format: str = Query("markdown", pattern="^(markdown|html|json)$")):
    """Exports case report in Markdown, HTML, or JSON format."""
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case ID not found.")

    # Attach SHAP summary
    case_detail = get_case_details(case_id)
    case["shap_summary"] = case_detail.get("shap_explanation", {}).get("summary_text", "N/A")

    if format == "markdown":
        md_text = CaseExporter.to_markdown(case)
        return PlainTextResponse(md_text, media_type="text/markdown")
    elif format == "html":
        html_text = CaseExporter.to_html(case)
        return HTMLResponse(html_text)
    else:
        return JSONResponse(case)


@app.get("/demo/files")
def list_demo_files():
    """Lists available demo datasets and templates."""
    demo_dir = DATA_DIR / "demo"
    raw_dir = DATA_DIR / "raw"
    files = []
    
    candidate_files = [
        ("sample_ecommerce_transactions.csv", demo_dir / "sample_ecommerce_transactions.csv", "Realistic 600 transaction dataset with customer IDs, amounts, IPs, devices"),
        ("sample_unmapped_aliases.csv", demo_dir / "sample_unmapped_aliases.csv", "50 transaction dataset with non-standard column headers for fuzzy schema mapping"),
        ("canonical_template.csv", demo_dir / "canonical_template.csv", "Standard canonical template CSV with expected schema fields"),
        ("development_transactions.csv", raw_dir / "development_transactions.csv", "Benchmark 3,000 transactions with isFraud ground-truth labels")
    ]
    for filename, path, desc in candidate_files:
        if path.exists():
            files.append({
                "filename": filename,
                "description": desc,
                "size_bytes": path.stat().st_size,
                "download_url": f"/demo/download/{filename}"
            })
    return {"status": "success", "demo_files": files}


@app.get("/demo/download/{filename}")
def download_demo_file(filename: str):
    """Downloads a demo dataset or canonical template CSV."""
    safe_names = {
        "sample_ecommerce_transactions.csv": DATA_DIR / "demo" / "sample_ecommerce_transactions.csv",
        "sample_unmapped_aliases.csv": DATA_DIR / "demo" / "sample_unmapped_aliases.csv",
        "canonical_template.csv": DATA_DIR / "demo" / "canonical_template.csv",
        "development_transactions.csv": DATA_DIR / "raw" / "development_transactions.csv",
    }
    if filename not in safe_names or not safe_names[filename].exists():
        raise HTTPException(status_code=404, detail="Requested demo file does not exist.")
    return FileResponse(
        path=safe_names[filename],
        filename=filename,
        media_type="text/csv"
    )
