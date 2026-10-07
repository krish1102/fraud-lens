---
document_id: TYP-002
title: "Fraud Typology: Card Testing and BIN Attacks"
document_type: fraud_typology
jurisdiction: Global / Online Payments
source: "Payment Systems Risk Group"
source_url: "https://fraudlens.internal/typologies/card-testing"
publication_date: "2024-04-10"
ingestion_date: "2024-10-01"
tags: [card_testing, bin_attack, micro_transactions, high_velocity, decline_rate]
---

# Typology Overview: Card Testing & BIN Attacks

## 1. Description
Card testing is an automated fraud technique where criminals systematically test batches of stolen payment credentials (PAN, expiration, CVV) by running small, rapid authorizations against an e-commerce checkout or donation endpoint. Once a card passes authorization, fraudsters exploit it for larger ticket purchases or sell the validated credentials on illicit marketplaces.

## 2. Key Signals & Signatures
- **Micro-Amount Pattern**: Transactions clustered around nominal values ($0.50, $1.00, $2.00, or INR 10-50).
- **Extreme Burst Velocity**: `transactions_last_5m >= 3` and `transactions_last_30m >= 8` across cards or within an analytical customer profile.
- **Card Churn**: Rapid succession of distinct card tokens or virtual cards originating from a single IP subnet or device fingerprint.
- **High Authorization Decline Ratio**: Substantial spike in pre-authorization declines followed by sporadic approvals.
- **Merchant Novelty**: Use of merchants with low-friction 3DS exemptions or instant digital delivery (gift cards, gaming credits).

## 3. Investigation Protocol
1. **Analyze Velocity Windows**: Examine 5-minute and 30-minute velocity metrics for the associated IP address and device identifier.
2. **Cluster Analysis**: Query the entity graph for other customer accounts or transaction IDs sharing the identical `device_id` or `/24` IP prefix.
3. **Issuer Pattern Review**: Identify if transactions share identical Bank Identification Numbers (BINs - first 6/8 digits of PAN).
4. **Action**: Implement rate-limiting, challenge with CAPTCHA or step-up authentication, and block repeatedly failing device fingerprints.
