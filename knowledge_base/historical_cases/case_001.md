---
document_id: HIST-001
title: "Synthetic Historical Benchmark Case: High-Velocity Account Takeover"
document_type: historical_case
jurisdiction: Benchmark / Synthetic (Not Real Historical PII)
source: "FraudLens Synthetic Benchmark Suite"
source_url: "https://fraudlens.internal/cases/case-001"
publication_date: "2024-08-01"
ingestion_date: "2024-10-01"
tags: [synthetic_case, account_takeover, new_device, velocity_burst, resolved_fraud]
---

# Benchmark Case #FRD-001052 (Synthetic Development Benchmark)
*Note: This case is a synthetic development case derived from benchmark data patterns for research and analytical demonstration. It contains no real personal data.*

## Case Metadata
- **Case ID**: FRD-001052
- **Transaction ID**: T98231
- **Analytical Customer**: C00102
- **Disputed Amount**: $1,840.50 (Customer Median: $65.00)
- **Model Scores**: Fraud Probability: 0.91 | Anomaly Score: 0.94 | Network Risk: 0.72 | Combined Risk: 0.88 (High)
- **Final Outcome**: Confirmed Fraudulent Account Takeover (Credentials compromised via phishing)

## Observed Signals & Timeline
1. At 02:14 UTC, a new device fingerprint (`D091`) initiated a login session from an unrecognized IP address (`203.0.113.10`).
2. Within 12 minutes, three sequential purchases were attempted:
   - T98229: $10.00 (Test charge at digital goods merchant)
   - T98230: $450.00 (Electronics store)
   - T98231: $1,840.50 (High-end luxury apparel)
3. SHAP local explanation attributed 31% positive weight to `amount_vs_customer_avg (+28.3x)`, 24% to `new_device = 1`, and 19% to `transactions_last_30m = 3`.
4. Entity graph analysis revealed device `D091` was also connected to analytical customer `C00205` which was flagged 2 days prior.

## Investigative Action Taken
- Level 1 analyst placed temporary settlement hold. Outbound SMS verification was sent to the registered customer phone number.
- Legitimate customer answered, denied knowledge of transaction, and stated they were asleep in another timezone.
- Account credentials invalidated, active tokens terminated, and dispute processed.
