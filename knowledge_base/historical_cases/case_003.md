---
document_id: HIST-003
title: "Synthetic Historical Benchmark Case: Shared Device Syndicate"
document_type: historical_case
jurisdiction: Benchmark / Synthetic (Not Real Historical PII)
source: "FraudLens Synthetic Benchmark Suite"
source_url: "https://fraudlens.internal/cases/case-003"
publication_date: "2024-08-20"
ingestion_date: "2024-10-01"
tags: [synthetic_case, syndicate, shared_device, mule_ring, high_degree]
---

# Benchmark Case #FRD-001140 (Synthetic Development Benchmark)
*Note: This case is a synthetic development case derived from benchmark data patterns for research and analytical demonstration. It contains no real personal data.*

## Case Metadata
- **Case ID**: FRD-001140
- **Transaction ID**: T88190
- **Analytical Customer**: C00902
- **Disputed Amount**: $780.00 (Customer Median: $120.00)
- **Model Scores**: Fraud Probability: 0.79 | Anomaly Score: 0.75 | Network Risk: 0.92 | Combined Risk: 0.81 (High)
- **Final Outcome**: Confirmed Multi-Account Device Ring / Promotion Abuse

## Observed Signals & Timeline
1. Hardware device fingerprint `D402` was utilized across 8 different customer profiles within 48 hours.
2. Network metrics showed: `device_customer_count = 8`, `suspicious_neighbor_count = 3`, `entity_degree = 9`.
3. High network risk score (0.92) drove the combined investigation priority into the High band.
4. Each connected account attempted rapid discount code redemption followed by immediate physical merchandise orders.

## Investigative Action Taken
- All 8 accounts linked to device `D402` were suspended.
- Orders in transit recalled from courier logistics partner.
- Syndicate graph structure exported to internal financial crimes risk register.
