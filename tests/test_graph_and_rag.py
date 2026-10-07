"""
Unit tests for Graph Engine and RAG Copilot.
"""

import pytest
import pandas as pd
from src.graph.graph_engine import GraphEngine
from src.rag.knowledge_indexer import KnowledgeIndexer
from src.rag.retriever import RagRetriever
from src.rag.investigation_copilot import InvestigationCopilot


def test_graph_engine_shared_devices():
    ge = GraphEngine()
    df = pd.DataFrame([
        {"customer_id": "C1", "device_id": "D_SHARED", "ip_address": "1.1.1.1"},
        {"customer_id": "C2", "device_id": "D_SHARED", "ip_address": "2.2.2.2"},
        {"customer_id": "C3", "device_id": "D_ISOLATED", "ip_address": "3.3.3.3"}
    ])
    ge.build_from_dataframe(df)

    metrics_c1 = ge.get_customer_network_metrics("C1")
    assert metrics_c1["shared_device_count"] == 1
    assert metrics_c1["connected_customers_count"] == 1

    metrics_c3 = ge.get_customer_network_metrics("C3")
    assert metrics_c3["shared_device_count"] == 0


def test_rag_retriever_and_citations():
    indexer = KnowledgeIndexer()
    assert indexer.load_index() is True

    retriever = RagRetriever(indexer)
    # Search account takeover
    ato_results = retriever.retrieve("account takeover new device credential stuffing", top_k=2)
    assert len(ato_results) > 0
    assert any("account_takeover" in r["document_id"] or "Takeover" in r["title"] for r in ato_results)

    # Search RBI regulatory
    rbi_results = retriever.retrieve("RBI Master Directions early warning signals", top_k=2)
    assert len(rbi_results) > 0
    assert any("RBI" in r["title"] or "REG" in r["document_id"] for r in rbi_results)
    assert rbi_results[0]["source_url"] != ""


def test_copilot_grounded_response():
    indexer = KnowledgeIndexer()
    indexer.load_index()
    retriever = RagRetriever(indexer)
    copilot = InvestigationCopilot(retriever)

    case_data = {
        "case_id": "FRD-TEST01",
        "transaction_id": "T999",
        "customer_id": "C888",
        "fraud_probability": 0.92,
        "anomaly_score": 0.88,
        "network_risk": 0.75,
        "risk_band": "HIGH",
        "detected_signals": ["new_device", "amount 5.0x above avg"]
    }

    resp = copilot.answer_query("Why was this transaction flagged?", case_data)
    assert "FRD-TEST01" in resp["answer"]
    assert "Supervised Fraud Probability" in resp["answer"]
    assert len(resp["sources"]) > 0
