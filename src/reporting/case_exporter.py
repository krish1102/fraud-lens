"""
Case Report Exporter:
Formats investigation cases into standardized Markdown, HTML, and JSON reports
matching the specification in Section 21.
"""

from typing import Dict, Any, List
import json


class CaseExporter:
    """Generates standardized investigation case reports."""

    @staticmethod
    def to_markdown(case: Dict[str, Any]) -> str:
        """
        Renders markdown report strictly following Section 21 Case Report Format.
        """
        case_id = case.get("case_id", "FRD-UNKNOWN")
        txn_id = case.get("transaction_id", "N/A")
        cust_id = case.get("customer_id", "N/A")
        mod_ver = case.get("model_version", "fraud_model_v1.2")
        p_fraud = case.get("fraud_probability", 0.0)
        s_anom = case.get("anomaly_score", 0.0)
        s_net = case.get("network_risk", 0.0)
        c_score = case.get("combined_score", 0.0)
        signals = case.get("detected_signals", [])
        if isinstance(signals, str):
            try:
                signals = json.loads(signals)
            except Exception:
                signals = [signals]

        shap_summary = case.get("shap_summary", "No SHAP explanation generated.")
        sources = case.get("rag_sources", "No regulatory documents cited.")
        rag_summary = case.get("rag_summary", "Preliminary automated triage complete.")
        notes = case.get("analyst_notes", "No notes recorded yet.")

        md = f"""# CASE #{case_id}
**Transaction**: `{txn_id}`  
**Analytical Customer**: `{cust_id}`  
**Model Version**: `{mod_ver}`  
**Status**: `{case.get('status', 'Open')}`  

---

### Scores
- **Fraud probability**: `{p_fraud:.2f}`
- **Anomaly score**: `{s_anom:.2f}`
- **Network risk**: `{s_net:.2f}`
- **Combined Priority**: `{c_score:.2f}` ({case.get('risk_band', 'MEDIUM')})

### Observed signals
"""
        for sig in signals:
            md += f"- {sig}\n"

        md += f"""
### SHAP explanation
```
{shap_summary}
```

### Relevant knowledge
{sources}

### RAG investigation summary
{rag_summary}

### Analyst notes
> {notes}
"""
        return md

    @staticmethod
    def to_html(case: Dict[str, Any]) -> str:
        """Renders an attractive, print-friendly HTML dossier."""
        md_text = CaseExporter.to_markdown(case)
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FraudLens Case #{case.get('case_id')}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f1f5f9; padding: 40px; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 30px; border: 1px solid #334155; max-width: 900px; margin: auto; }}
  h1 {{ color: #38bdf8; margin-top: 0; }}
  h3 {{ color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
  pre {{ background: #090d16; padding: 16px; border-radius: 8px; border: 1px solid #1e293b; color: #38bdf8; font-family: monospace; }}
  .badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-weight: bold; background: #ef4444; color: white; }}
  ul {{ line-height: 1.6; }}
  blockquote {{ border-left: 4px solid #38bdf8; margin-left: 0; padding-left: 16px; color: #cbd5e1; }}
</style>
</head>
<body>
<div class="card">
  <h1>CASE #{case.get('case_id')}</h1>
  <p><strong>Transaction:</strong> {case.get('transaction_id')} | <strong>Customer:</strong> {case.get('customer_id')} | <span class="badge">{case.get('risk_band')}</span></p>
  <h3>Model Risk Scores</h3>
  <p>Fraud Probability: <strong>{case.get('fraud_probability')}</strong> | Anomaly Score: <strong>{case.get('anomaly_score')}</strong> | Network Risk: <strong>{case.get('network_risk')}</strong></p>
  <h3>Observed Signals</h3>
  <ul>
    {"".join(f"<li>{s}</li>" for s in case.get('detected_signals', []))}
  </ul>
  <h3>SHAP Feature Explanation</h3>
  <pre>{case.get('shap_summary', 'N/A')}</pre>
  <h3>Analyst Notes</h3>
  <blockquote>{case.get('analyst_notes', 'None')}</blockquote>
</div>
</body>
</html>"""
        return html
