"""
Unit tests for supervised and anomaly models, risk engine, and SHAP explainer.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.supervised_model import SupervisedFraudModel
from src.models.anomaly_model import UnsupervisedAnomalyModel
from src.models.risk_engine import RiskEngine
from src.explainability.shap_explainer import ShapExplainer
from src.configs.settings import MODEL_FEATURES


def test_models_load_and_predict():
    sup_model = SupervisedFraudModel()
    loaded_sup = sup_model.load()
    assert loaded_sup is True

    anom_model = UnsupervisedAnomalyModel()
    loaded_anom = anom_model.load()
    assert loaded_anom is True

    # Dummy feature row
    dummy_data = {f: [1.0] for f in MODEL_FEATURES}
    dummy_df = pd.DataFrame(dummy_data)

    p_fraud = sup_model.predict_proba(dummy_df)
    assert len(p_fraud) == 1
    assert 0.0 <= p_fraud[0] <= 1.0

    s_anom = anom_model.score(dummy_df)
    assert len(s_anom) == 1
    assert 0.0 <= s_anom[0] <= 1.0


def test_risk_engine_scoring_and_signals():
    risk_engine = RiskEngine()
    combined = risk_engine.calculate_combined_score(
        fraud_probability=0.90,
        anomaly_score=0.80,
        network_risk=0.70
    )
    assert 0.0 <= combined <= 1.0
    band = risk_engine.get_risk_band(combined)
    assert band == "HIGH"

    test_row = pd.Series({
        "amount_vs_customer_avg": 4.5,
        "new_device": 1,
        "customer_transaction_count": 5,
        "transactions_last_30m": 4,
        "device_customer_count": 4
    })
    signals = risk_engine.extract_detected_signals(test_row)
    assert any("higher than customer average" in s for s in signals)
    assert any("New/unrecognized hardware device" in s for s in signals)


def test_shap_explainer_local_attribution():
    sup_model = SupervisedFraudModel()
    sup_model.load()

    explainer = ShapExplainer(sup_model.model)
    dummy_row = pd.Series({f: 1.0 for f in MODEL_FEATURES})
    res = explainer.explain_instance(dummy_row, fraud_probability=0.85)

    assert "summary_text" in res
    assert "Fraud probability: 0.85" in res["summary_text"]
    assert "Top contributing signals:" in res["summary_text"]
    assert "They do not by themselves prove fraudulent activity." in res["summary_text"]
