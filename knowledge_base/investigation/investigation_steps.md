---
document_id: INV-001
title: "Standard Investigation Procedure: FraudLens Analyst SOP"
document_type: investigation_guidance
jurisdiction: Global / E-Commerce Operations
source: "FraudLens Security Operations Center"
source_url: "https://fraudlens.internal/sop/investigation-steps"
publication_date: "2024-05-15"
ingestion_date: "2024-10-01"
tags: [investigation, sop, triage, evidence_review, resolution]
---

# Fraud Analyst Standard Operating Procedure (SOP)

## 1. Initial Case Intake & Triage
When a transaction is flagged with a high combined risk score (`combined_risk >= 0.70`), the analyst must perform triage within 15 minutes:
1. **Review Risk Decomposition**:
   - Inspect `fraud_probability` (Supervised model - pattern match against historical frauds).
   - Inspect `anomaly_score` (Unsupervised Isolation Forest - deviation from typical behaviors).
   - Inspect `network_risk` (Graph degree and shared malicious infrastructure).
2. **Examine SHAP Local Explanations**:
   - Identify the top 3-5 positive contributors driving the risk score.
   - Distinguish quantitative model evidence from business facts (SHAP reveals why the model flagged it, not absolute proof of fraud).

## 2. Customer 360 & Baseline Comparison
1. Compare transaction amount against `customer_avg_amount` and `customer_median_amount`.
2. Inspect `customer_amount_zscore`: If z-score > 3.0, the transaction is an extreme statistical outlier.
3. Check `time_since_previous_txn` and recent 30-minute velocity.
4. Review prior device and merchant familiarity.

## 3. Entity Graph Inspection
1. Navigate to the Fraud Network explorer for the case.
2. Check if the customer's device or IP address is linked to any other customer (`shared_device_count`, `shared_ip_count`).
3. Check `suspicious_neighbor_count`: If any neighboring entity has an existing open or resolved fraud case, mark as high syndicate probability.

## 4. Decisioning & Action
- **Approve**: Legitimate behavior change (e.g. holiday gift purchase with consistent device, confirmed out-of-band).
- **Challenge / Step-Up**: Moderate risk, new device/IP but customer has established history; challenge with SMS OTP or biometric verification.
- **Decline & Freeze**: High risk, confirmed ATO, stolen card, or multi-account device cluster. Add customer notes and export case report.
