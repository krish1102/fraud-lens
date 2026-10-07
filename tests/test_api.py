"""
API integration tests for FastAPI application endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["system"] == "FraudLens"
    assert data["status"] == "Ready"


def test_api_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "cases_overview" in data
    assert data["rag_kb_chunks"] > 0


def test_api_upload_and_pipeline_flow():
    csv_content = """transaction_id,customer_id,timestamp,amount,device_id,merchant_id,ip_address
T101,C001,2026-09-01 10:00:00,50.00,D01,M01,1.1.1.1
T102,C001,2026-09-01 10:05:00,500.00,D02,M02,2.2.2.2
"""
    # 1. Upload
    files = {"file": ("test.csv", csv_content, "text/csv")}
    upload_res = client.post("/upload", files=files)
    assert upload_res.status_code == 200
    ds_id = upload_res.json()["dataset_id"]

    # 2. Analyze
    analyze_res = client.post(f"/schema/analyze?dataset_id={ds_id}")
    assert analyze_res.status_code == 200
    assert "suggestions" in analyze_res.json()

    # 3. Validate
    val_res = client.post(f"/validate?dataset_id={ds_id}")
    assert val_res.status_code == 200
    assert val_res.json()["validation_report"]["is_valid"] is True

    # 4. Predict
    pred_res = client.post(f"/predict?dataset_id={ds_id}")
    assert pred_res.status_code == 200
    assert pred_res.json()["total_scored"] == 2

    # 5. Retrieve cases
    cases_res = client.get("/cases")
    assert cases_res.status_code == 200
    cases_data = cases_res.json()
    assert isinstance(cases_data["cases"], list)

    if cases_data["cases"]:
        first_case_id = cases_data["cases"][0]["case_id"]
        # 6. Case details
        detail_res = client.get(f"/case/{first_case_id}")
        assert detail_res.status_code == 200
        assert "shap_explanation" in detail_res.json()

        # 7. Case Chat
        chat_res = client.post(f"/case/{first_case_id}/chat", json={"query": "Why was this flagged?"})
        assert chat_res.status_code == 200
        assert "answer" in chat_res.json()

        # 8. Export Case
        export_res = client.get(f"/export/{first_case_id}?format=markdown")
        assert export_res.status_code == 200
        assert "# CASE #" in export_res.text
