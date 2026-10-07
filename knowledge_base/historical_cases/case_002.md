---
document_id: HIST-002
title: "Synthetic Historical Benchmark Case: Micro-Charge Card Testing Attack"
document_type: historical_case
jurisdiction: Benchmark / Synthetic (Not Real Historical PII)
source: "FraudLens Synthetic Benchmark Suite"
source_url: "https://fraudlens.internal/cases/case-002"
publication_date: "2024-08-10"
ingestion_date: "2024-10-01"
tags: [synthetic_case, card_testing, micro_charges, rapid_velocity, botnet]
---

# Benchmark Case #FRD-001089 (Synthetic Development Benchmark)
*Note: This case is a synthetic development case derived from benchmark data patterns for research and analytical demonstration. It contains no real personal data.*

## Case Metadata
- **Case ID**: FRD-001089
- **Transaction ID**: T76412
- **Analytical Customer**: C00418
- **Disputed Amount**: $1.25 (Customer Median: $42.00)
- **Model Scores**: Fraud Probability: 0.84 | Anomaly Score: 0.89 | Network Risk: 0.81 | Combined Risk: 0.85 (High)
- **Final Outcome**: Confirmed Automated Script / Card Testing Botnet

## Observed Signals & Timeline
1. Single IP address (`198.51.100.42`) sent 14 transactions within 4 minutes across 14 distinct payment cards.
2. Velocity counters registered: `transactions_last_5m = 9`, `transactions_last_30m = 14`.
3. Entity degree for the IP node surged to 14 connected customers within one hour (`ip_customer_count = 14`).
4. Isolation Forest anomaly score was elevated due to extreme velocity burst and tiny amounts.

## Investigative Action Taken
- IP address was blacklisted at the WAF level.
- Automated rule applied to trigger CAPTCHA challenges on checkouts originating from that ASN.
- Affected issuing banks notified regarding compromised card numbers.
