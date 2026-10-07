---
document_id: TYP-003
title: "Fraud Typology: Velocity Anomalies & Bust-Out Exploitation"
document_type: fraud_typology
jurisdiction: Global / E-Commerce
source: "Fraud Strategy Research Group"
source_url: "https://fraudlens.internal/typologies/velocity-anomaly"
publication_date: "2024-05-02"
ingestion_date: "2024-10-01"
tags: [velocity, rapid_fire, bust_out, credit_drain, temporal_anomaly]
---

# Typology Overview: Velocity Anomalies & Bust-Out Exploitation

## 1. Description
Velocity anomalies occur when transaction volume or cumulative spending accelerates drastically over a brief temporal window compared to the established baseline. In bust-out fraud, a compromised account or artificially aged profile is suddenly exhausted to maximum credit limits before detection mechanisms can intervene.

## 2. Key Signals
- **Drastic Rate-of-Spend Increase**: `amount_last_30m` exceeding 500% of historical weekly average spend.
- **Inter-Transaction Gap Compression**: `time_since_previous_txn` dropping from normal days/hours to seconds or minutes (< 120 seconds).
- **Time Bucket Shift**: Multiple transactions occurring during unusual hours (`is_night = 1` or high `hour_deviation_from_customer_norm`).
- **Sequential Merchant Draining**: Transactions executed across multiple high-risk or instant liquidity merchants in rapid succession.

## 3. Recommended Checks
1. Calculate the rolling 5m, 30m, and 24h count and cumulative amount.
2. Cross-reference customer historical baseline (median amount and normal daily limits).
3. If velocity spikes co-occur with a `new_device` or `new_ip`, escalate immediately to priority queue.
