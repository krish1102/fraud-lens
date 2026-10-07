"""
FraudLens Analyst Dashboard
Customer-Centric Fraud & Anomaly Detection with RAG-Powered Investigation
"""

import sys
from pathlib import Path
import json
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.settings import (
    CANONICAL_FIELDS,
    COLUMN_ALIASES,
    MODEL_FEATURES,
    FEATURE_VERSION,
    MODEL_VERSION,
    DATA_DIR,
    MODELS_DIR
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

# Page Configuration
st.set_page_config(
    page_title="FraudLens | AI Fraud & Anomaly Investigation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Styling - Clean, decluttered, modern slate palette
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #0B0F19;
        color: #F1F5F9;
    }
    
    /* Top Stepper Breadcrumb */
    .stepper-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 10px 16px;
        margin-bottom: 20px;
        overflow-x: auto;
    }
    .stepper-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 500;
        color: #64748B;
        white-space: nowrap;
        padding: 4px 10px;
        border-radius: 8px;
    }
    .stepper-item.active {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        font-weight: 600;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    .stepper-item.completed {
        color: #10B981;
    }
    .stepper-sep {
        color: #334155;
        font-size: 12px;
    }
    
    /* Hero / Page Header Banner */
    .page-banner {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .page-title {
        font-size: 24px;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .page-desc {
        color: #94A3B8;
        font-size: 13.5px;
        margin: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .metric-title {
        font-size: 11.5px;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-num {
        font-size: 22px;
        font-weight: 700;
        color: #F8FAFC;
    }
    
    /* Risk Badges */
    .badge-high {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-low {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    
    /* Footer Navigation Bar */
    .nav-footer-box {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 14px 20px;
        margin-top: 30px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* Chat Bubbles */
    .chat-bubble-user {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .chat-bubble-assistant {
        background: #0F172A;
        border: 1px solid #0284C7;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.15);
    }
    
    /* Compact demo cards */
    .demo-grid-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 10px;
        padding: 12px 14px;
        height: 100%;
    }
    .demo-grid-title {
        font-size: 13.5px;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 3px;
    }
    .demo-grid-desc {
        font-size: 11.5px;
        color: #94A3B8;
        line-height: 1.35;
        margin-bottom: 10px;
    }
    
    /* Clean sidebar section */
    .sidebar-header {
        font-size: 11px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 14px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# Lazy-loaded singletons in Streamlit session state
@st.cache_resource
def get_services():
    db = DatabaseManager()
    feature_engine = TimeAwareFeatureEngine()
    
    sup_model = SupervisedFraudModel()
    sup_model.load()
    
    anom_model = UnsupervisedAnomalyModel()
    anom_model.load()
    
    graph_engine = GraphEngine()
    shap_explainer = ShapExplainer(sup_model.model)
    
    indexer = KnowledgeIndexer()
    if not indexer.load_index():
        indexer.build_index()
        
    retriever = RagRetriever(indexer)
    copilot = InvestigationCopilot(retriever)
    risk_engine = RiskEngine()
    
    return {
        "db": db,
        "feature_engine": feature_engine,
        "supervised_model": sup_model,
        "anomaly_model": anom_model,
        "graph_engine": graph_engine,
        "shap_explainer": shap_explainer,
        "retriever": retriever,
        "copilot": copilot,
        "risk_engine": risk_engine,
        "indexer": indexer
    }

services = get_services()


# Network Graph 2D Plotly Helper
def build_network_figure(subgraph, center_id: str, title: str = "Entity Linkage Network"):
    if not subgraph or not subgraph.get("nodes"):
        return None
    
    nodes = subgraph["nodes"]
    edges = subgraph["edges"]
    
    center_key = None
    for n in nodes:
        if n["node_id"] == center_id or n["id"] == f"cust:{center_id}" or n["id"] == center_id:
            center_key = n["id"]
            break
            
    other_nodes = [n for n in nodes if n["id"] != center_key]
    pos = {}
    if center_key:
        pos[center_key] = (0.0, 0.0)
    
    num_others = len(other_nodes)
    for idx, n in enumerate(other_nodes):
        angle = 2 * np.pi * idx / max(num_others, 1)
        r = 1.0
        pos[n["id"]] = (r * np.cos(angle), r * np.sin(angle))
        
    edge_x = []
    edge_y = []
    for e in edges:
        u, v = e["source"], e["target"]
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="rgba(148, 163, 184, 0.4)"),
        hoverinfo="none",
        mode="lines"
    )
    
    type_color_map = {
        "customer": "#38BDF8",     # Sky Blue
        "device": "#2DD4BF",       # Teal/Cyan
        "ip": "#F59E0B",           # Amber
        "merchant": "#A855F7",     # Purple
        "beneficiary": "#EC4899"   # Pink
    }
    type_symbol_map = {
        "customer": "circle",
        "device": "square",
        "ip": "diamond",
        "merchant": "circle",
        "beneficiary": "star"
    }
    
    fig = go.Figure()
    fig.add_trace(edge_trace)
    
    for ntype, color in type_color_map.items():
        group_nodes = [n for n in nodes if n.get("type", "unknown") == ntype and n["id"] in pos]
        if not group_nodes:
            continue
            
        xs = [pos[n["id"]][0] for n in group_nodes]
        ys = [pos[n["id"]][1] for n in group_nodes]
        texts = [
            f"<b>{n['label']}</b><br>Type: {n['type'].upper()}<br>ID: {n['node_id']}<br>Connected Links: {n['degree']}"
            for n in group_nodes
        ]
        sizes = [24 if n["id"] == center_key else 16 for n in group_nodes]
        
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            text=[n['label'] if n['id'] == center_key else "" for n in group_nodes],
            textposition="top center",
            hoverinfo="text",
            hovertext=texts,
            name=ntype.capitalize(),
            marker=dict(
                color=color,
                size=sizes,
                symbol=type_symbol_map.get(ntype, "circle"),
                line=dict(width=1.5, color="#FFFFFF")
            )
        ))
        
    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.5)",
        margin=dict(t=40, b=20, l=20, r=20),
        height=380
    )
    return fig


# ------------------ WORKFLOW STEPS DEFINITION ------------------
WORKFLOW_STEPS = [
    {"label": "1. 🏠 Data Ingestion", "short": "Upload", "name": "Data Ingestion & Demo Hub"},
    {"label": "2. 🔍 Schema Audit", "short": "Schema", "name": "Schema Mapping & Quality Audit"},
    {"label": "3. 📊 Executive Overview", "short": "Overview", "name": "Executive Operations Dashboard"},
    {"label": "4. ⚡ Transaction Monitor", "short": "Monitor", "name": "Real-Time Transaction Risk Monitor"},
    {"label": "5. 🛡️ Case Dossier", "short": "Case Dossier", "name": "Investigation Case Workspace"},
    {"label": "6. 🌐 Network Graph", "short": "Network", "name": "Entity Linkage & Fraud Ring Graph"},
    {"label": "7. 👤 Customer 360", "short": "Customer 360", "name": "Customer 360 & Behavioral Profile"},
    {"label": "8. 📚 Regulatory RAG", "short": "RAG Copilot", "name": "Regulatory Knowledge Base & Copilot"}
]
STEP_LABELS = [s["label"] for s in WORKFLOW_STEPS]

# Session State Initialization
if "current_df" not in st.session_state:
    st.session_state["current_df"] = None
if "scored_df" not in st.session_state:
    st.session_state["scored_df"] = None
if "validation_report" not in st.session_state:
    st.session_state["validation_report"] = None
if "selected_case_id" not in st.session_state:
    st.session_state["selected_case_id"] = None
if "active_step_idx" not in st.session_state:
    st.session_state["active_step_idx"] = 0


# Navigation Functions
def jump_to_step(idx: int):
    st.session_state["active_step_idx"] = idx


def render_stepper(current_idx: int):
    """Renders a sleek, responsive step progression bar at top of page."""
    html_parts = ['<div class="stepper-container">']
    for i, s in enumerate(WORKFLOW_STEPS):
        cls = "stepper-item"
        icon = "✓" if (i < current_idx and st.session_state["scored_df"] is not None) else f"{i+1}."
        if i == current_idx:
            cls += " active"
        elif i < current_idx:
            cls += " completed"
        html_parts.append(f'<div class="{cls}"><span>{icon}</span> {s["short"]}</div>')
        if i < len(WORKFLOW_STEPS) - 1:
            html_parts.append('<span class="stepper-sep">→</span>')
    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str, icon: str = "🛡️"):
    st.markdown(f"""
    <div class="page-banner">
        <div class="page-title">{icon} {title}</div>
        <p class="page-desc">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_step_footer(current_idx: int):
    """Renders bottom Previous / Next Step workflow controls."""
    st.markdown("---")
    c_prev, c_center, c_next = st.columns([1.5, 2, 1.5])
    
    with c_prev:
        if current_idx > 0:
            prev_label = WORKFLOW_STEPS[current_idx - 1]["short"]
            if st.button(f"⬅ Previous: {prev_label}", key=f"foot_prev_{current_idx}", width="stretch"):
                jump_to_step(current_idx - 1)
                st.rerun()
                
    with c_center:
        st.markdown(
            f"<div style='text-align:center; color:#94A3B8; font-size:13px; padding-top:8px;'>"
            f"Step <b>{current_idx + 1}</b> of <b>8</b>: {WORKFLOW_STEPS[current_idx]['name']}</div>",
            unsafe_allow_html=True
        )
        
    with c_next:
        if current_idx < len(WORKFLOW_STEPS) - 1:
            next_label = WORKFLOW_STEPS[current_idx + 1]["short"]
            if st.button(f"Next: {next_label} ➔", type="primary", key=f"foot_next_{current_idx}", width="stretch"):
                jump_to_step(current_idx + 1)
                st.rerun()


def chart_tip(title: str, text: str):
    """Collapsible clean explanation for any chart to eliminate UI clutter."""
    with st.expander(f"💡 How to read this chart: {title}"):
        st.markdown(text)


# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.markdown("### 🛡️ **FraudLens**")
    st.caption(f"Enterprise ML Engine `{MODEL_VERSION}` | Schema `{FEATURE_VERSION}`")
    
    # Dataset Status Badge
    if st.session_state["scored_df"] is not None:
        st.success(f"● {len(st.session_state['scored_df'])} Txns Scored & Ready")
    elif st.session_state["current_df"] is not None:
        st.info(f"● {len(st.session_state['current_df'])} Txns Ingested (Unscored)")
    else:
        st.warning("● No Dataset Active")
        
    st.markdown("<div class='sidebar-header'>Investigation Workflow</div>", unsafe_allow_html=True)
    
    current_step = st.session_state.get("active_step_idx", 0)
    st.session_state["sidebar_workflow_nav"] = STEP_LABELS[current_step]
    
    def on_sidebar_nav_change():
        chosen = st.session_state.get("sidebar_workflow_nav")
        if chosen in STEP_LABELS:
            st.session_state["active_step_idx"] = STEP_LABELS.index(chosen)

    st.radio(
        "Workflow Steps",
        STEP_LABELS,
        key="sidebar_workflow_nav",
        on_change=on_sidebar_nav_change,
        label_visibility="collapsed"
    )

    st.markdown("---")
    
    # Clean Collapsible Demo Datasets & Downloads
    with st.expander("⚡ Demo Datasets & Templates", expanded=False):
        st.caption("Quickly test detection pipelines:")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("Load 600 Txns", key="sb_load_600", width="stretch", help="Load 600 transactions into active session"):
                sample_path = DATA_DIR / "demo" / "sample_ecommerce_transactions.csv"
                if sample_path.exists():
                    st.session_state["current_df"] = pd.read_csv(sample_path)
                    st.session_state["scored_df"] = None
                    st.session_state["validation_report"] = None
                    jump_to_step(0)
                    st.success("Loaded 600 demo transactions!")
                    st.rerun()
        with col_s2:
            if st.button("Load Aliases", key="sb_load_alias", width="stretch", help="Load CSV with non-standard column headers"):
                alias_path = DATA_DIR / "demo" / "sample_unmapped_aliases.csv"
                if alias_path.exists():
                    st.session_state["current_df"] = pd.read_csv(alias_path)
                    st.session_state["scored_df"] = None
                    st.session_state["validation_report"] = None
                    jump_to_step(0)
                    st.info("Loaded unmapped alias CSV!")
                    st.rerun()
                    
        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        # Download buttons
        sample_path = DATA_DIR / "demo" / "sample_ecommerce_transactions.csv"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                st.download_button("📥 Sample 600 Txns (CSV)", data=f.read(), file_name="sample_ecommerce_transactions.csv", mime="text/csv", width="stretch")
        alias_path = DATA_DIR / "demo" / "sample_unmapped_aliases.csv"
        if alias_path.exists():
            with open(alias_path, "rb") as f:
                st.download_button("📥 Unmapped Aliases (CSV)", data=f.read(), file_name="sample_unmapped_aliases.csv", mime="text/csv", width="stretch")
        template_path = DATA_DIR / "demo" / "canonical_template.csv"
        if template_path.exists():
            with open(template_path, "rb") as f:
                st.download_button("📥 Canonical Template", data=f.read(), file_name="canonical_transaction_template.csv", mime="text/csv", width="stretch")

    if st.session_state["current_df"] is not None:
        st.download_button(
            "💾 Export Current CSV",
            data=st.session_state["current_df"].to_csv(index=False).encode("utf-8"),
            file_name="fraudlens_active_dataset.csv",
            mime="text/csv",
            width="stretch"
        )
        
    if st.button("🔄 Reset Session", width="stretch", help="Clear active data and restart investigation"):
        st.session_state["current_df"] = None
        st.session_state["scored_df"] = None
        st.session_state["validation_report"] = None
        st.session_state["selected_case_id"] = None
        jump_to_step(0)
        st.rerun()


# Active Step Index
current_step = st.session_state["active_step_idx"]

# Render Stepper at Top of Main View
render_stepper(current_step)


# ==============================================================================
# 1. 🏠 DATA INGESTION & DEMO HUB
# ==============================================================================
if current_step == 0:
    render_page_header(
        "Data Ingestion & Demo Hub",
        "Ingest online transaction streams via custom CSV upload or test instantly with benchmark datasets.",
        "🏠"
    )
    
    # Demo dataset quick-cards
    st.markdown("#### ⚡ 1-Click Demo Datasets")
    d1, d2, d3, d4 = st.columns(4)
    
    sample_path = DATA_DIR / "demo" / "sample_ecommerce_transactions.csv"
    with d1:
        st.markdown("""
        <div class="demo-grid-card">
            <div class="demo-grid-title">🛒 600 E-Commerce Txns</div>
            <div class="demo-grid-desc">Clean, ready-to-score dataset with customer IDs, amounts, IPs, and device fingerprints.</div>
        </div>
        """, unsafe_allow_html=True)
        col_b1, col_b2 = st.columns(2)
        if sample_path.exists():
            with col_b1:
                with open(sample_path, "rb") as f:
                    st.download_button("📥 CSV", data=f.read(), file_name="sample_ecommerce_transactions.csv", mime="text/csv", key="h_dl_600", width="stretch")
            with col_b2:
                if st.button("⚡ Load", key="h_ld_600", type="primary", width="stretch"):
                    st.session_state["current_df"] = pd.read_csv(sample_path)
                    st.session_state["scored_df"] = None
                    st.session_state["validation_report"] = None
                    st.success("Loaded 600 transactions!")
                    st.rerun()

    alias_path = DATA_DIR / "demo" / "sample_unmapped_aliases.csv"
    with d2:
        st.markdown("""
        <div class="demo-grid-card">
            <div class="demo-grid-title">🔀 Unmapped Aliases (50)</div>
            <div class="demo-grid-desc">Non-standard column headers (<code>cust_no</code>, <code>txn_val</code>) to test fuzzy auto-mapping.</div>
        </div>
        """, unsafe_allow_html=True)
        col_b1, col_b2 = st.columns(2)
        if alias_path.exists():
            with col_b1:
                with open(alias_path, "rb") as f:
                    st.download_button("📥 CSV", data=f.read(), file_name="sample_unmapped_aliases.csv", mime="text/csv", key="h_dl_alias", width="stretch")
            with col_b2:
                if st.button("⚡ Load", key="h_ld_alias", width="stretch"):
                    st.session_state["current_df"] = pd.read_csv(alias_path)
                    st.session_state["scored_df"] = None
                    st.session_state["validation_report"] = None
                    st.info("Loaded unmapped alias CSV!")
                    st.rerun()

    template_path = DATA_DIR / "demo" / "canonical_template.csv"
    with d3:
        st.markdown("""
        <div class="demo-grid-card">
            <div class="demo-grid-title">📋 Canonical Blueprint</div>
            <div class="demo-grid-desc">Standard schema template showing expected column definitions and formats.</div>
        </div>
        """, unsafe_allow_html=True)
        if template_path.exists():
            with open(template_path, "rb") as f:
                st.download_button("📥 Download Template", data=f.read(), file_name="canonical_transaction_template.csv", mime="text/csv", key="h_dl_tpl", width="stretch")

    dev_path = DATA_DIR / "raw" / "development_transactions.csv"
    with d4:
        st.markdown("""
        <div class="demo-grid-card">
            <div class="demo-grid-title">🏷️ 3K Labeled Benchmark</div>
            <div class="demo-grid-desc">Full 3,000 transaction dataset with ground-truth <code>isFraud</code> targets.</div>
        </div>
        """, unsafe_allow_html=True)
        if dev_path.exists():
            with open(dev_path, "rb") as f:
                st.download_button("📥 Download Benchmark", data=f.read(), file_name="development_transactions.csv", mime="text/csv", key="h_dl_dev", width="stretch")

    st.markdown("---")
    
    # Custom File Uploader + Status Area
    col_up, col_schema = st.columns([3, 2])
    with col_up:
        st.subheader("Upload Custom Transaction CSV")
        uploaded_file = st.file_uploader(
            "Upload transaction CSV (canonical format or arbitrary column names)",
            type=["csv"]
        )
        if uploaded_file is not None:
            try:
                df, meta = CSVLoader.load_csv(uploaded_file.getvalue())
                st.session_state["current_df"] = df
                st.session_state["scored_df"] = None
                st.session_state["validation_report"] = None
                st.success(f"Loaded '{uploaded_file.name}' with {len(df)} records!")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
                
        if st.session_state["current_df"] is not None:
            st.success(f"✅ **Active Dataset**: `{len(st.session_state['current_df'])}` rows ready for schema mapping & validation.")
            st.dataframe(st.session_state["current_df"].head(6), width="stretch", hide_index=True)
            
            # Primary Call to Action to go to Step 2
            if st.button("➡️ Proceed to Step 2: Schema & Quality Audit", type="primary", key="go_to_schema_btn", width="stretch"):
                jump_to_step(1)
                st.rerun()

    with col_schema:
        st.subheader("Expected Canonical Schema")
        schema_data = [
            {"Field": k, "Required": "Yes" if v["required"] else "No", "Type": v["type"], "Description": v["description"]}
            for k, v in CANONICAL_FIELDS.items()
        ]
        st.dataframe(pd.DataFrame(schema_data), hide_index=True, width="stretch", height=240)
        st.caption("💡 Known aliases (`txn_id`, `cust_no`, `ip`, `device`) map automatically.")
        
    render_step_footer(0)


# ==============================================================================
# 2. 🔍 SCHEMA AUDIT & DATA QUALITY
# ==============================================================================
elif current_step == 1:
    render_page_header(
        "Schema Mapping & Quality Audit",
        "Confirm header alignments, validate completeness, and execute dual-model ML feature scoring.",
        "🔍"
    )
    
    if st.session_state["current_df"] is None:
        st.warning("⚠️ No active dataset found. Please load or upload transactions in Step 1 first.")
        if st.button("⬅ Return to Step 1: Data Ingestion", type="primary"):
            jump_to_step(0)
            st.rerun()
    else:
        df_raw = st.session_state["current_df"]
        profile = SchemaAnalyzer.analyze(df_raw)
        mapper = ColumnMapper()
        suggestions = mapper.suggest_mappings(list(df_raw.columns))
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows Ingested", profile["total_rows"])
        m2.metric("Columns Detected", profile["total_columns"])
        m3.metric("Mapped Canonical Fields", len(suggestions["mapped"]))
        
        st.markdown("#### 1. Confirm Column Mapping")
        st.caption("Verify that source columns correspond correctly to canonical fraud features.")
        
        confirmed_mapping = {}
        mapping_cols = st.columns(4)
        for idx, (canonical, spec) in enumerate(CANONICAL_FIELDS.items()):
            c_col = mapping_cols[idx % 4]
            suggested_src = suggestions["mapped"].get(canonical, None)
            options = ["-- Unmapped --"] + list(df_raw.columns)
            default_idx = options.index(suggested_src) if suggested_src in options else 0
            req_tag = "*(Required)*" if spec["required"] else "*(Optional)*"
            
            chosen = c_col.selectbox(
                f"{canonical} {req_tag}",
                options=options,
                index=default_idx,
                key=f"map_{canonical}"
            )
            if chosen != "-- Unmapped --":
                confirmed_mapping[canonical] = chosen

        st.markdown("---")
        st.markdown("#### 2. Quality Audit & Analytics Capabilities")
        
        col_act1, col_act2 = st.columns([1, 1])
        with col_act1:
            if st.button("Run Schema Audit & Quality Validation", type="secondary", width="stretch"):
                canonical_df = ColumnMapper.apply_mapping(df_raw, confirmed_mapping)
                cleaned_df, report = DataValidator.validate_and_clean(canonical_df)
                st.session_state["validation_report"] = report
                if report["is_valid"]:
                    st.session_state["current_df"] = cleaned_df
                    st.success("Validation passed!")
                    st.rerun()
                    
        if st.session_state["validation_report"] is not None:
            rep = st.session_state["validation_report"]
            if rep["is_valid"]:
                st.success("✅ **Dataset Certified**: Conforms to canonical requirements and ready for time-aware scoring.")
            else:
                st.error("❌ Dataset failed validation checks:")
                for r in rep["block_reasons"]:
                    st.write(f"- {r}")
                    
            if rep["warnings"]:
                with st.expander("Validation Warnings / Type Coercions", expanded=False):
                    for w in rep["warnings"]:
                        st.info(f"⚠️ {w}")
                        
            cap = rep["capabilities"]
            b1, b2, b3 = st.columns(3)
            b1.markdown(f"**Basic Detection**: {'🟢 Full' if cap['basic'] else '🔴 Missing Fields'}")
            b2.markdown(f"**Behavioral Baseline**: {'🟢 Enabled' if cap['behavioral'] else '🟡 Partial'}")
            b3.markdown(f"**Entity Linkage Graph**: {'🟢 Enabled' if cap['network'] else '🟡 Partial'}")
            
            if rep["is_valid"]:
                st.markdown("---")
                st.markdown("#### 3. Execute Dual-Model ML Scoring")
                
                # Check if already scored
                if st.session_state["scored_df"] is not None:
                    st.success(f"✅ **Scoring Active**: {len(st.session_state['scored_df'])} transactions scored and ready.")
                    if st.button("➡️ Proceed to Step 3: Executive Overview", type="primary", key="to_ov_btn", width="stretch"):
                        jump_to_step(2)
                        st.rerun()
                else:
                    if st.button("🚀 Run Time-Aware Feature Engine & Score Dataset", type="primary", width="stretch"):
                        with st.spinner("Calculating velocity features, running Supervised & Anomaly models..."):
                            enriched, feat_mat = services["feature_engine"].build_features(st.session_state["current_df"])
                            p_fraud = services["supervised_model"].predict_proba(feat_mat)
                            s_anom = services["anomaly_model"].score(feat_mat)
                            
                            services["graph_engine"].build_from_dataframe(enriched)
                            net_risks = []
                            for _, row in enriched.iterrows():
                                nm = services["graph_engine"].get_customer_network_metrics(str(row["customer_id"]))
                                net_risks.append(nm["network_risk"])
                                
                            comb_scores = []
                            bands = []
                            signals = []
                            for i in range(len(enriched)):
                                cs = services["risk_engine"].calculate_combined_score(p_fraud[i], s_anom[i], net_risks[i])
                                comb_scores.append(cs)
                                bands.append(services["risk_engine"].get_risk_band(cs))
                                signals.append(services["risk_engine"].extract_detected_signals(enriched.iloc[i]))
                                
                            scored = enriched.copy()
                            scored["fraud_probability"] = p_fraud
                            scored["anomaly_score"] = s_anom
                            scored["network_risk"] = net_risks
                            scored["combined_score"] = comb_scores
                            scored["risk_band"] = bands
                            scored["detected_signals"] = signals
                            
                            st.session_state["scored_df"] = scored
                            
                            records = []
                            for _, r in scored.iterrows():
                                records.append({
                                    "transaction_id": str(r["transaction_id"]),
                                    "customer_id": str(r["customer_id"]),
                                    "fraud_probability": r["fraud_probability"],
                                    "anomaly_score": r["anomaly_score"],
                                    "network_risk": r["network_risk"],
                                    "combined_score": r["combined_score"],
                                    "risk_band": r["risk_band"],
                                    "detected_signals": r["detected_signals"]
                                })
                            services["db"].save_predictions_and_cases(records)
                            st.success(f"Scored {len(scored)} transactions!")
                            st.rerun()
                            
    render_step_footer(1)


# ==============================================================================
# 3. 📊 EXECUTIVE OVERVIEW
# ==============================================================================
elif current_step == 2:
    render_page_header(
        "Executive Operations Dashboard",
        "Portfolio health, risk tier distributions, financial exposure, and dual-model intelligence.",
        "📊"
    )
    
    if st.session_state["scored_df"] is None:
        st.info("No scored dataset in memory. Please complete scoring in Step 2 or load demo data below.")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            if st.button("⚡ Load & Auto-Score 600 Demo Transactions", type="primary", width="stretch"):
                sample_path = DATA_DIR / "demo" / "sample_ecommerce_transactions.csv"
                if sample_path.exists():
                    df = pd.read_csv(sample_path)
                    st.session_state["current_df"] = df
                    enriched, feat_mat = services["feature_engine"].build_features(df)
                    p_fraud = services["supervised_model"].predict_proba(feat_mat)
                    s_anom = services["anomaly_model"].score(feat_mat)
                    services["graph_engine"].build_from_dataframe(enriched)
                    net_risks = [services["graph_engine"].get_customer_network_metrics(str(r["customer_id"]))["network_risk"] for _, r in enriched.iterrows()]
                    comb_scores = [services["risk_engine"].calculate_combined_score(p_fraud[i], s_anom[i], net_risks[i]) for i in range(len(enriched))]
                    bands = [services["risk_engine"].get_risk_band(cs) for cs in comb_scores]
                    signals = [services["risk_engine"].extract_detected_signals(enriched.iloc[i]) for i in range(len(enriched))]
                    scored = enriched.copy()
                    scored["fraud_probability"] = p_fraud
                    scored["anomaly_score"] = s_anom
                    scored["network_risk"] = net_risks
                    scored["combined_score"] = comb_scores
                    scored["risk_band"] = bands
                    scored["detected_signals"] = signals
                    st.session_state["scored_df"] = scored
                    records = [{"transaction_id": str(r["transaction_id"]), "customer_id": str(r["customer_id"]), "fraud_probability": r["fraud_probability"], "anomaly_score": r["anomaly_score"], "network_risk": r["network_risk"], "combined_score": r["combined_score"], "risk_band": r["risk_band"], "detected_signals": r["detected_signals"]} for _, r in scored.iterrows()]
                    services["db"].save_predictions_and_cases(records)
                    st.rerun()
        with col_q2:
            if st.button("⬅ Go to Step 2: Schema & Quality Audit", width="stretch"):
                jump_to_step(1)
                st.rerun()
    else:
        df = st.session_state["scored_df"]
        total_txns = len(df)
        high_df = df[df["risk_band"] == "HIGH"]
        med_df = df[df["risk_band"] == "MEDIUM"]
        low_df = df[df["risk_band"] == "LOW"]
        
        high_cnt = len(high_df)
        med_cnt = len(med_df)
        low_cnt = len(low_df)
        total_vol = df["amount"].sum()
        at_risk_vol = high_df["amount"].sum()
        
        # High-impact metric cards
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Transactions", f"{total_txns:,}")
        k2.metric("Total Processed Volume", f"${total_vol:,.2f}")
        k3.metric("High-Risk Alerts", f"{high_cnt}", f"{high_cnt/total_txns*100:.1f}%")
        k4.metric("Medium-Risk Triage", f"{med_cnt}", f"{med_cnt/total_txns*100:.1f}%")
        k5.metric("At-Risk Volume", f"${at_risk_vol:,.2f}")
        
        st.markdown("---")
        
        # Visual charts
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("1. Risk Priority Band Breakdown")
            fig_pie = px.pie(
                values=[high_cnt, med_cnt, low_cnt],
                names=["High Risk (Immediate Block)", "Medium Risk (Step-Up Triage)", "Low Risk (Auto-Approved)"],
                color_discrete_sequence=["#EF4444", "#F59E0B", "#10B981"],
                hole=0.45
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_pie, width="stretch")
            
            chart_tip(
                "Risk Priority Tiers",
                "• **High Risk (Red, ≥0.70)**: High probability fraud. Requires immediate blocking or card freeze.<br>"
                "• **Medium Risk (Amber, 0.40–0.69)**: Suspicious signals. Route to analyst queue or step-up 2FA.<br>"
                "• **Low Risk (Green, <0.40)**: Normal genuine customer activity. Seamless auto-approval."
            )

        with c2:
            st.subheader("2. Dual-AI Radar: Known Patterns vs. Zero-Day Anomalies")
            fig_scatter = px.scatter(
                df,
                x="fraud_probability",
                y="anomaly_score",
                color="risk_band",
                size="amount",
                hover_data=["transaction_id", "customer_id", "amount"],
                color_discrete_map={"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"},
                labels={
                    "fraud_probability": "Supervised AI Score (Known Patterns)",
                    "anomaly_score": "Isolation Forest Score (Zero-Day Outliers)",
                    "risk_band": "Risk Tier"
                }
            )
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.5)",
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_scatter, width="stretch")
            
            chart_tip(
                "Dual-Model AI Coordinates",
                "• **Top-Right (High Supervised + High Anomaly)**: Definite attack with abnormal traits.<br>"
                "• **Top-Left (Zero-Day Anomaly)**: Novel fraud pattern missed by supervised models but caught by Isolation Forest.<br>"
                "• **Bubble Size**: Dollar volume ($). Large bubbles represent high financial liability."
            )

        st.markdown("---")
        st.subheader("3. Transaction Timeline & Burst Attack Detection")
        fig_time = px.scatter(
            df,
            x="timestamp",
            y="amount",
            color="combined_score",
            color_continuous_scale="Reds",
            hover_data=["transaction_id", "customer_id", "risk_band"],
            labels={
                "timestamp": "Timestamp",
                "amount": "Transaction Amount ($)",
                "combined_score": "Risk Score (0.0 to 1.0)"
            }
        )
        fig_time.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.5)",
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_time, width="stretch")
        
        chart_tip(
            "Temporal Outlier Map",
            "• **Dense Clusters of Red Dots**: Automated bot attacks or card testing attacks hitting in a short window.<br>"
            "• **Tall Red Spikes**: Large fraudulent payments attempting account drain.<br>"
            "• **Grey/Light Dots**: Routine business activity."
        )
        
    render_step_footer(2)


# ==============================================================================
# 4. ⚡ TRANSACTION MONITOR
# ==============================================================================
elif current_step == 3:
    render_page_header(
        "Real-Time Transaction Risk Monitor",
        "Filter, search, and triage active transaction alerts. Select any case to initiate formal investigation.",
        "⚡"
    )
    
    if st.session_state["scored_df"] is None:
        st.info("No scored dataset available. Please complete scoring in Step 2 or load demo data in Step 3.")
    else:
        df = st.session_state["scored_df"]
        
        # Clean Filters
        f1, f2, f3 = st.columns([1, 1, 2])
        with f1:
            band_filter = st.multiselect("Filter Risk Tier", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
        with f2:
            min_score = st.slider("Minimum Risk Score", 0.0, 1.0, 0.40, 0.05)
        with f3:
            search_query = st.text_input("Search Transaction ID or Customer ID", placeholder="e.g., CUST-1042 or T1004")
            
        filtered = df[df["risk_band"].isin(band_filter) & (df["combined_score"] >= min_score)]
        if search_query:
            filtered = filtered[
                filtered["transaction_id"].str.contains(search_query, case=False) |
                filtered["customer_id"].str.contains(search_query, case=False)
            ]
            
        st.caption(f"Showing **{len(filtered)}** prioritized transactions out of **{len(df)}** total:")
        
        display_cols = [
            "transaction_id", "customer_id", "timestamp", "amount",
            "fraud_probability", "anomaly_score", "network_risk", "combined_score", "risk_band"
        ]
        
        st.dataframe(
            filtered[display_cols].sort_values(by="combined_score", ascending=False),
            width="stretch",
            hide_index=True
        )
        
        st.markdown("---")
        st.subheader("🚀 Open Case Dossier")
        st.caption("Select a suspicious transaction to inspect its SHAP explainability, entity linkages, and RAG Copilot guidance.")
        
        col_sel, col_action = st.columns([3, 2])
        with col_sel:
            candidate_ids = filtered["transaction_id"].tolist() if not filtered.empty else ["No matching transactions"]
            selected_tid = st.selectbox("Select Transaction ID", candidate_ids, label_visibility="collapsed")
            
        with col_action:
            if selected_tid and selected_tid != "No matching transactions":
                case_id = f"FRD-{selected_tid.lstrip('T').zfill(6)}"
                if st.button(f"🔍 Investigate Case {case_id} in Step 5 ➔", type="primary", width="stretch"):
                    st.session_state["selected_case_id"] = case_id
                    jump_to_step(4)
                    st.rerun()
                    
    render_step_footer(3)


# ==============================================================================
# 5. 🛡️ CASE INVESTIGATION
# ==============================================================================
elif current_step == 4:
    render_page_header(
        "Investigation Case Workspace & RAG Copilot",
        "Deep-dive into flagged payments with SHAP explainability, entity graphs, and regulatory AI guidance.",
        "🛡️"
    )
    
    cases = services["db"].get_all_cases()
    if not cases:
        st.info("No active investigation cases recorded. Please score a dataset in Step 2 first.")
    else:
        case_options = [c["case_id"] for c in cases]
        default_case_idx = 0
        if st.session_state["selected_case_id"] in case_options:
            default_case_idx = case_options.index(st.session_state["selected_case_id"])
            
        c_sel1, c_sel2 = st.columns([3, 1])
        with c_sel1:
            chosen_case_id = st.selectbox("Active Case Dossier", case_options, index=default_case_idx)
            st.session_state["selected_case_id"] = chosen_case_id
        with c_sel2:
            st.caption("Switch case anytime")
            
        case = services["db"].get_case(chosen_case_id)
        if case:
            tid = case["transaction_id"]
            cid = case["customer_id"]
            
            row_data = None
            if st.session_state["scored_df"] is not None:
                matches = st.session_state["scored_df"][st.session_state["scored_df"]["transaction_id"] == tid]
                if not matches.empty:
                    row_data = matches.iloc[0]
                    
            # Summary Metrics Row
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Case Reference", case["case_id"])
            h2.metric("Target Customer", cid)
            h3.metric("Transaction ID", tid)
            h4.metric("Risk Priority", f"{case['combined_score']:.2f}", f"Tier: {case['risk_band']}")
            
            # Scores Breakout
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Supervised Model", f"{case['fraud_probability']:.2f}")
            sc2.metric("Anomaly Score", f"{case['anomaly_score']:.2f}")
            sc3.metric("Network Risk", f"{case['network_risk']:.2f}")
            sc4.metric("Status", case.get("status", "Open"))
            
            st.markdown("---")
            
            signals = case.get("detected_signals", [])
            if isinstance(signals, str):
                try:
                    signals = json.loads(signals)
                except Exception:
                    signals = [signals]
                    
            t_signals, t_shap, t_network, t_rag, t_report = st.tabs([
                "🚨 Behavioral Signals",
                "📈 SHAP Explainability",
                "🕸️ Network Evidence",
                "🤖 RAG Copilot",
                "📄 Export & Audit"
            ])
            
            # TAB 1: SIGNALS
            with t_signals:
                st.subheader("Observed Behavioral & Operational Signals")
                for s in signals:
                    st.markdown(f"- 🔴 **{s}**")
                    
                if row_data is not None:
                    st.markdown("#### Transaction Attribute Comparison")
                    comp_df = pd.DataFrame({
                        "Metric": [
                            "Current Amount", "Customer Avg Amount", "Customer Median Amount",
                            "Amount / Avg Ratio", "Customer Z-Score", "Prior 30m Txn Count",
                            "Time Since Last Txn", "New Device Flag"
                        ],
                        "Value": [
                            f"${row_data.get('transaction_amount', 0):.2f}",
                            f"${row_data.get('customer_avg_amount', 0):.2f}",
                            f"${row_data.get('customer_median_amount', 0):.2f}",
                            f"{row_data.get('amount_vs_customer_avg', 1.0):.2f}x",
                            f"{row_data.get('customer_amount_zscore', 0):.2f}",
                            f"{int(row_data.get('transactions_last_30m', 0))}",
                            f"{row_data.get('time_since_previous_txn', -1):.1f}s",
                            "Yes (First Time)" if row_data.get('new_device', 0) == 1 else "No (Known Device)"
                        ]
                    })
                    st.table(comp_df)

            # TAB 2: SHAP
            with t_shap:
                st.subheader("AI Decision Explainability (SHAP Feature Contributions)")
                if row_data is not None:
                    shap_res = services["shap_explainer"].explain_instance(row_data, case["fraud_probability"])
                    
                    chart_tip(
                        "SHAP Feature Attribution",
                        "• **Horizontal Red Bars**: Features that specifically elevated the fraud probability score for this payment.<br>"
                        "• **Interpretation**: Shows exactly which behavioral anomalies (e.g. 5x baseline spike, velocity surge) influenced the AI model."
                    )
                    
                    top_pos = shap_res["top_positive"]
                    if top_pos:
                        friendly_names = {
                            "amount_vs_customer_avg": "Amount vs Customer Average",
                            "transactions_last_30m": "High Velocity (Last 30 Min)",
                            "new_device": "First Time Device Observed",
                            "customer_amount_zscore": "Spending Deviation Z-Score",
                            "time_since_previous_txn": "Short Time Gap Since Prior Txn",
                            "ip_customer_count": "Multiple Accounts On This IP",
                            "device_customer_count": "Multiple Accounts On This Device",
                            "amount_vs_merchant_avg": "Amount vs Merchant Average",
                            "hourly_tx_density": "Hourly Transaction Spike",
                            "distance_from_home": "Unusual Location Distance"
                        }
                        feat_names = [friendly_names.get(p["feature"], p["feature"].replace("_", " ").title()) for p in top_pos]
                        contributions = [round(p["contribution"], 3) for p in top_pos]
                        
                        fig_shap = go.Figure(go.Bar(
                            x=contributions,
                            y=feat_names,
                            orientation="h",
                            marker=dict(color="#EF4444"),
                            text=[f"+{c:.3f}" for c in contributions],
                            textposition="outside"
                        ))
                        fig_shap.update_layout(
                            title="Key Drivers Elevating Fraud Risk (+ Points)",
                            xaxis_title="SHAP Attribution (+ Risk Points)",
                            yaxis=dict(autorange="reversed"),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(17,24,39,0.5)",
                            margin=dict(t=35, b=20, l=20, r=50)
                        )
                        st.plotly_chart(fig_shap, width="stretch")
                    else:
                        st.success("No dominant anomalous features identified.")
                else:
                    st.info("Feature record not available for this case.")

            # TAB 3: NETWORK
            with t_network:
                st.subheader(f"Entity Linkage Graph — Customer `{cid}`")
                net_metrics = services["graph_engine"].get_customer_network_metrics(cid)
                subgraph = services["graph_engine"].get_subgraph_data(cid, max_hops=2)
                
                nc1, nc2, nc3 = st.columns(3)
                nc1.metric("Shared Devices", net_metrics["shared_device_count"])
                nc2.metric("Shared IPs", net_metrics["shared_ip_count"])
                nc3.metric("Suspicious Connected Entities", net_metrics["suspicious_neighbor_count"])
                st.info(f"**Assessment**: {net_metrics['summary']}")
                
                fig_net = build_network_figure(subgraph, cid, title=f"Ego Network for Customer {cid}")
                if fig_net is not None:
                    st.plotly_chart(fig_net, width="stretch")
                else:
                    st.write("Customer is an isolated node with no cross-entity linkages.")
                    
                chart_tip(
                    "Entity Linkage Evidence",
                    "• **Blue**: Target Customer.<br>"
                    "• **Teal**: Connected Devices.<br>"
                    "• **Amber**: Connected IP Addresses.<br>"
                    "• **Multi-Account Collisions**: If multiple accounts share a phone or IP, it suggests organized syndicates."
                )
                
                if st.button("🌐 Inspect Full Fraud Network in Step 6 ➔", key="to_step_6_btn"):
                    jump_to_step(5)
                    st.rerun()

            # TAB 4: RAG COPILOT
            with t_rag:
                st.subheader("RAG Regulatory & Typology Copilot")
                st.caption("Ask questions grounded in official RBI circulars, fraud typologies, and analyst SOPs.")
                
                chat_key = f"copilot_messages_{chosen_case_id}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                    
                def ask_copilot(query_text: str):
                    if not query_text or not query_text.strip():
                        return
                    case_payload = {
                        "case_id": case["case_id"],
                        "transaction_id": case["transaction_id"],
                        "customer_id": case["customer_id"],
                        "fraud_probability": case["fraud_probability"],
                        "anomaly_score": case["anomaly_score"],
                        "network_risk": case["network_risk"],
                        "risk_band": case["risk_band"],
                        "detected_signals": signals
                    }
                    if row_data is not None:
                        shap_info = services["shap_explainer"].explain_instance(row_data, case["fraud_probability"])
                        case_payload["shap_summary"] = shap_info["summary_text"]
                        
                    with st.spinner("Analyzing case evidence & searching regulatory knowledge base..."):
                        resp = services["copilot"].answer_query(query_text, case_payload)
                        st.session_state[chat_key].append({
                            "query": query_text,
                            "answer": resp["answer"],
                            "sources": resp.get("sources", []),
                            "intent": resp.get("intent", "general")
                        })
                        
                st.markdown("**⚡ 1-Click Investigation Questions:**")
                q1, q2, q3, q4, q5 = st.columns(5)
                if q1.button("Why Flagged?", width="stretch"):
                    ask_copilot("Why was this transaction flagged?")
                    st.rerun()
                if q2.button("Fraud Typology?", width="stretch"):
                    ask_copilot("What fraud pattern does this resemble?")
                    st.rerun()
                if q3.button("Next Steps?", width="stretch"):
                    ask_copilot("What should an analyst check next?")
                    st.rerun()
                if q4.button("RBI Circular?", width="stretch"):
                    ask_copilot("What does RBI say about early warning signals and fraud risk management?")
                    st.rerun()
                if q5.button("Similar Cases?", width="stretch"):
                    ask_copilot("Show similar benchmark cases")
                    st.rerun()
                    
                st.markdown("---")
                with st.form(key=f"copilot_form_{chosen_case_id}", clear_on_submit=True):
                    col_in, col_sub = st.columns([5, 1])
                    with col_in:
                        custom_q = st.text_input("Custom Investigation Question", placeholder="e.g. Is this pattern consistent with mule accounts?", label_visibility="collapsed")
                    with col_sub:
                        submitted = st.form_submit_button("Ask Copilot 🚀", width="stretch")
                    if submitted and custom_q:
                        ask_copilot(custom_q)
                        st.rerun()
                        
                if st.session_state[chat_key]:
                    col_ch1, col_ch2 = st.columns([5, 1])
                    with col_ch1:
                        st.caption(f"Conversation Thread ({len(st.session_state[chat_key])} exchanges):")
                    with col_ch2:
                        if st.button("Clear Chat", key=f"clr_{chosen_case_id}", width="stretch"):
                            st.session_state[chat_key] = []
                            st.rerun()
                            
                    for turn in reversed(st.session_state[chat_key]):
                        st.markdown(f'<div class="chat-bubble-user"><b>Analyst:</b> {turn["query"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="chat-bubble-assistant"><b>🛡️ FraudLens Copilot:</b><div style="margin-top:6px;">{turn["answer"]}</div></div>', unsafe_allow_html=True)
                        if turn.get("sources"):
                            with st.expander(f"📚 Grounded Knowledge Sources ({len(turn['sources'])} cited documents)"):
                                for s in turn["sources"]:
                                    url_link = f" • [Official Source Link]({s['source_url']})" if s.get("source_url") else ""
                                    st.markdown(
                                        f"**`[{s['document_id']}]` {s['title']}**\n\n"
                                        f"*Section*: `{s.get('section_title', 'Overview')}` | *Jurisdiction*: `{s.get('jurisdiction', 'Global')}` | *Publisher*: `{s.get('source', 'Internal Policy')}`{url_link}"
                                    )
                                    st.markdown("---")

            # TAB 5: REPORT
            with t_report:
                st.subheader("Case Notes & Export Dossier")
                current_notes = st.text_area("Analyst Disposition Notes:", value=case.get("analyst_notes") or "")
                new_status = st.selectbox("Update Case Status:", ["Open", "Investigating", "Escalated", "Resolved - Confirmed Fraud", "Resolved - False Positive"], index=0)
                
                if st.button("Save Case Notes", type="secondary"):
                    services["db"].update_case_notes(chosen_case_id, current_notes, new_status)
                    st.success("Case notes saved to SQLite registry!")
                    
                st.markdown("---")
                case_for_export = dict(case)
                case_for_export["analyst_notes"] = current_notes
                case_for_export["status"] = new_status
                if row_data is not None:
                    shap_info = services["shap_explainer"].explain_instance(row_data, case["fraud_probability"])
                    case_for_export["shap_summary"] = shap_info["summary_text"]
                    
                ex1, ex2 = st.columns(2)
                with ex1:
                    st.download_button("📄 Download Markdown Report", data=CaseExporter.to_markdown(case_for_export), file_name=f"{chosen_case_id}_report.md", mime="text/markdown", width="stretch")
                with ex2:
                    st.download_button("🌐 Download HTML Dossier", data=CaseExporter.to_html(case_for_export), file_name=f"{chosen_case_id}_dossier.html", mime="text/html", width="stretch")

    render_step_footer(4)


# ==============================================================================
# 6. 🌐 FRAUD NETWORK GRAPH
# ==============================================================================
elif current_step == 5:
    render_page_header(
        "Heterogeneous Fraud & Entity Network Explorer",
        "Uncover shared physical infrastructure, device farms, and high-frequency IP hubs.",
        "🌐"
    )
    
    if st.session_state["scored_df"] is None:
        st.info("No network data in memory. Please run scoring in Step 2 first.")
    else:
        df = st.session_state["scored_df"]
        top_devs = df["device_id"].value_counts().head(5)
        top_ips = df["ip_address"].value_counts().head(5)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top Shared Devices (Hardware Rings)")
            fig_dev = px.bar(
                x=top_devs.values,
                y=top_devs.index,
                orientation='h',
                labels={"x": "Total Transactions", "y": "Device Fingerprint ID"},
                color=top_devs.values,
                color_continuous_scale="Teal"
            )
            fig_dev.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.5)",
                showlegend=False,
                margin=dict(t=10, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_dev, width="stretch")
            
            chart_tip(
                "Shared Device Hubs",
                "• Devices with transactions across multiple distinct customer accounts indicate emulator farms or device cloning."
            )

        with c2:
            st.subheader("Top Active IP Addresses (Network Hubs)")
            fig_ip = px.bar(
                x=top_ips.values,
                y=top_ips.index,
                orientation='h',
                labels={"x": "Total Transactions", "y": "IP Address"},
                color=top_ips.values,
                color_continuous_scale="Oranges"
            )
            fig_ip.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.5)",
                showlegend=False,
                margin=dict(t=10, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_ip, width="stretch")
            
            chart_tip(
                "IP Concentration",
                "• Heavy transaction bursts from a single IP often point to commercial VPNs, proxies, or automated credential bots."
            )

        st.markdown("---")
        st.subheader("Customer Ego Network Visualizer")
        search_cust = st.selectbox("Select Customer to Visualize", df["customer_id"].unique()[:50])
        
        if search_cust:
            subgraph = services["graph_engine"].get_subgraph_data(search_cust, max_hops=2)
            metrics = services["graph_engine"].get_customer_network_metrics(search_cust)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Shared Devices", metrics["shared_device_count"])
            m2.metric("Shared IPs", metrics["shared_ip_count"])
            m3.metric("Connected Accounts", metrics.get("connected_customers_count", 0))
            m4.metric("Network Risk", f"{metrics['network_risk']:.2f}")
            
            fig_ego = build_network_figure(subgraph, search_cust, title=f"Visual Connection Graph for {search_cust}")
            if fig_ego:
                st.plotly_chart(fig_ego, width="stretch")
                
            st.info(f"**Assessment**: {metrics['summary']}")
            
            if st.button(f"👤 View {search_cust} in Step 7 (Customer 360) ➔"):
                jump_to_step(6)
                st.rerun()

    render_step_footer(5)


# ==============================================================================
# 7. 👤 CUSTOMER 360
# ==============================================================================
elif current_step == 6:
    render_page_header(
        "Customer 360 & Behavioral Profile",
        "Examine individual customer spending timelines, baseline deviations, and historical transactions.",
        "👤"
    )
    
    if st.session_state["scored_df"] is None:
        st.info("No customer data in memory. Please run scoring first.")
    else:
        df = st.session_state["scored_df"]
        cust_list = df["customer_id"].unique().tolist()
        selected_cust = st.selectbox("Select Customer Profile", cust_list)
        
        cust_txns = df[df["customer_id"] == selected_cust].sort_values(by="timestamp")
        avg_amt = float(cust_txns['amount'].mean())
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Observed Txns", len(cust_txns))
        k2.metric("Cumulative Spend", f"${cust_txns['amount'].sum():.2f}")
        k3.metric("Baseline Average", f"${avg_amt:.2f}")
        k4.metric("Highest Alert Tier", cust_txns["risk_band"].max())
        
        st.markdown("---")
        st.subheader("Spending Timeline vs. Historical Baseline")
        
        fig_cust = px.scatter(
            cust_txns,
            x="timestamp",
            y="amount",
            color="risk_band",
            color_discrete_map={"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"},
            hover_data=["transaction_id", "risk_band", "combined_score"],
            labels={"timestamp": "Transaction Time", "amount": "Amount ($)", "risk_band": "Risk Tier"}
        )
        fig_cust.add_trace(go.Scatter(
            x=cust_txns["timestamp"],
            y=cust_txns["amount"],
            mode="lines",
            line=dict(color="rgba(148, 163, 184, 0.3)", width=1.5),
            hoverinfo="none",
            showlegend=False
        ))
        fig_cust.add_hline(
            y=avg_amt,
            line_dash="dash",
            line_color="#38BDF8",
            annotation_text=f"Baseline (${avg_amt:.2f})",
            annotation_position="top left",
            annotation_font_color="#38BDF8"
        )
        fig_cust.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.5)",
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_cust, width="stretch")
        
        chart_tip(
            "Customer Spending Velocity",
            "• **Dashed Blue Line**: Customer's typical average purchase amount.<br>"
            "• **Red/Amber Dots**: Sudden high-value transactions deviating significantly from their baseline."
        )
        
        st.dataframe(
            cust_txns[["transaction_id", "timestamp", "amount", "merchant_id", "device_id", "risk_band", "combined_score"]],
            width="stretch",
            hide_index=True
        )
        
    render_step_footer(6)


# ==============================================================================
# 8. 📚 REGULATORY RAG KNOWLEDGE BASE
# ==============================================================================
elif current_step == 7:
    render_page_header(
        "Regulatory Knowledge Base & RAG Copilot",
        "Official regulatory circulars (RBI Master Directions), fraud typologies, SOPs, and benchmark cases.",
        "📚"
    )
    
    chunks = services["indexer"].chunks
    st.caption(f"Knowledge Repository contains **{len(chunks)}** indexed chunks across 4 analytical categories.")
    
    col_k1, col_k2 = st.columns([1, 2])
    with col_k1:
        filter_type = st.selectbox(
            "Filter Category",
            ["All", "fraud_typologies", "investigation", "regulatory", "historical_cases"],
            format_func=lambda x: {
                "All": "🌐 All Categories (56 Chunks)",
                "fraud_typologies": "🚨 Fraud Typologies",
                "investigation": "📋 Investigation SOPs",
                "regulatory": "🏛️ Regulatory Circulars (RBI / Fed)",
                "historical_cases": "📁 Historical Benchmark Cases"
            }.get(x, x)
        )
    with col_k2:
        search_term = st.text_input(
            "Search Knowledge Base (Keywords or Questions)",
            placeholder="e.g., mule accounts, RBI early warning signals, card testing...",
            key="rag_search_input"
        )
        
    st.markdown("**⚡ 1-Click Example Searches:**")
    pill_cols = st.columns(5)
    selected_quick = None
    if pill_cols[0].button("Card Testing", width="stretch"):
        selected_quick = "card testing bin attack"
    if pill_cols[1].button("Mule Accounts", width="stretch"):
        selected_quick = "mule accounts money laundering"
    if pill_cols[2].button("RBI Guidance", width="stretch"):
        selected_quick = "RBI early warning signals fraud risk management"
    if pill_cols[3].button("Account Takeover", width="stretch"):
        selected_quick = "account takeover credential stuffing"
    if pill_cols[4].button("Analyst SOP", width="stretch"):
        selected_quick = "standard investigation procedure analyst sop"
        
    active_search = selected_quick or search_term
    
    if active_search and active_search.strip():
        st.markdown(f"#### 🔍 Search Results for: *'{active_search}'*")
        doc_filter = None if filter_type == "All" else filter_type
        retrieved_results = services["retriever"].retrieve(
            active_search,
            top_k=6,
            doc_type=doc_filter,
            min_score=0.02,
            deduplicate_docs=False
        )
        
        if retrieved_results:
            st.caption(f"Retrieved {len(retrieved_results)} relevant knowledge sections ranked by semantic & keyword match:")
            for r in retrieved_results:
                match_pct = int(min(max(r["score"] * 250, 45), 98))
                with st.expander(f"📌 [{r['document_id']}] {r['title']} — {r.get('section_title', 'Overview')} (Match: {match_pct}%)", expanded=True):
                    mc1, mc2 = st.columns([3, 1])
                    with mc1:
                        st.markdown(f"**Document ID**: `{r['document_id']}` | **Category**: `{r['document_type']}` | **Jurisdiction**: `{r['jurisdiction']}`")
                        if r.get("source_url"):
                            st.markdown(f"**Official Source**: [{r['source_url']}]({r['source_url']})")
                    with mc2:
                        st.markdown(f"`{', '.join(r.get('tags', [])[:3])}`")
                    st.markdown("---")
                    st.markdown(r["text"])
        else:
            query_words = set(active_search.lower().split())
            fallback_chunks = [
                c for c in chunks
                if (filter_type == "All" or c.document_type == filter_type) and
                any(len(w) > 3 and (w in c.title.lower() or w in c.text.lower()) for w in query_words)
            ]
            if fallback_chunks:
                st.caption(f"Found {len(fallback_chunks)} chunks with keyword overlap:")
                for c in fallback_chunks[:6]:
                    with st.expander(f"📌 [{c.document_id}] {c.title} — {c.section_title}"):
                        st.markdown(f"**Category**: `{c.document_type}` | **Jurisdiction**: `{c.jurisdiction}`")
                        if c.source_url:
                            st.markdown(f"**Source**: [{c.source_url}]({c.source_url})")
                        st.markdown(c.text)
            else:
                st.warning(f"No direct matches found for *'{active_search}'*. Showing canonical documents in category **{filter_type}** below:")
                docs = services["indexer"].get_document_catalog(filter_type)
                for doc in docs[:8]:
                    with st.expander(f"📂 [{doc['document_id']}] {doc['title']} ({len(doc['sections'])} Sections)"):
                        st.markdown(f"**Category**: `{doc['document_type']}` | **Jurisdiction**: `{doc['jurisdiction']}` | **Publisher**: `{doc['source']}`")
                        if doc.get("source_url"):
                            st.markdown(f"**Official URL**: [{doc['source_url']}]({doc['source_url']})")
                        for s in doc["sections"]:
                            st.markdown(f"#### {s['section_title']}")
                            st.markdown(s["text"])
                            st.markdown("---")
    else:
        # Default browse view: Group by distinct canonical documents!
        docs = services["indexer"].get_document_catalog(filter_type)
        cat_label = filter_type.replace('_', ' ').title() if filter_type != "All" else "All Categories"
        st.markdown(f"#### 📂 Repository Documents: `{cat_label}` ({len(docs)} Distinct Documents)")
        for doc in docs:
            with st.expander(f"📘 [{doc['document_id']}] {doc['title']} — {doc['jurisdiction']} ({len(doc['sections'])} Sections)", expanded=False):
                m_c1, m_c2 = st.columns([3, 1])
                with m_c1:
                    st.markdown(f"**Category**: `{doc['document_type']}` | **Jurisdiction**: `{doc['jurisdiction']}`")
                    st.markdown(f"**Publisher / Source**: `{doc['source']}` ({doc['publication_date']})")
                    if doc.get("source_url"):
                        st.markdown(f"**Official Link**: [{doc['source_url']}]({doc['source_url']})")
                with m_c2:
                    st.caption("Document Tags:")
                    st.markdown(f"`{', '.join(doc['tags'])}`")
                    
                st.markdown("---")
                for s in doc["sections"]:
                    st.markdown(f"##### {s['section_title']}")
                    st.markdown(s["text"])
                    st.markdown("---")
                
    render_step_footer(7)
