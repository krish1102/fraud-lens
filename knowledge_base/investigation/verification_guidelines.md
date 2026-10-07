---
document_id: INV-002
title: "Customer Verification & Identity Challenge Guidelines"
document_type: investigation_guidance
jurisdiction: Global / E-Commerce Operations
source: "Identity Trust & Risk Policy Group"
source_url: "https://fraudlens.internal/sop/verification-guidelines"
publication_date: "2024-06-01"
ingestion_date: "2024-10-01"
tags: [verification, step_up, otp, mfa, out_of_band, kyc]
---

# Customer Verification & Step-Up Authentication Guidelines

## 1. Scope & Objective
Guidance on when and how an analyst or automated pipeline should prompt a customer for additional identity verification when abnormal signals are detected.

## 2. Decision Matrix
- **Tier 1 (Passive Approval)**:
  - Low fraud probability (< 0.40) AND low anomaly score (< 0.40).
  - Existing trusted device, consistent IP subnet, transaction within 1.5x customer median.
  - Action: Frictionless checkout.

- **Tier 2 (Step-Up Challenge - MFA / OTP)**:
  - Moderate risk (0.40 - 0.70) OR single isolated novelty signal (`new_device = 1` or `new_ip = 1`).
  - Action: Issue out-of-band one-time passcode (OTP) or push notification to registered mobile application.

- **Tier 3 (Manual Analyst Contact & Hold)**:
  - High risk (>= 0.70) OR multiple compound signals (e.g., `new_device = 1` + `amount_vs_customer_avg > 3.0` + `transactions_last_30m > 3`).
  - Action: Temporary hold on fulfillment; outbound voice verification or document upload request.

- **Tier 4 (Immediate Revocation & Blacklist)**:
  - Known stolen credential list, confirmed mule node in entity graph, or high hub degree with prior confirmed fraud cases.
  - Action: Immediate rejection, terminate active auth tokens, log to blacklisted entity registry.
