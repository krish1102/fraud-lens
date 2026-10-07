---
document_id: INV-003
title: "Fraud Escalation Matrix & Law Enforcement Reporting"
document_type: investigation_guidance
jurisdiction: Global / Financial Crimes Unit
source: "FraudLens Incident Response Policy"
source_url: "https://fraudlens.internal/sop/escalation-guidelines"
publication_date: "2024-06-15"
ingestion_date: "2024-10-01"
tags: [escalation, incident_response, law_enforcement, sar, strata]
---

# Fraud Escalation & Reporting Matrix

## 1. Internal Escalation Triggers
An investigation case must be escalated from Level 1 Analyst to Level 2 / Fraud Manager under the following criteria:
1. **Cumulative Exposure**: Total disputed or flagged transaction volume associated with the customer or device entity exceeds $5,000 (or INR 2,50,000).
2. **Syndicate / Network Activity**: Entity graph reveals that 3 or more distinct analytical customer IDs share the same hardware device or IP address with active high-risk flags.
3. **Internal Compromise / Systematic Exploit**: Suspected coupon abuse or platform vulnerability exploitation.

## 2. Regulatory Reporting & SAR (Suspicious Activity Report)
- When a confirmed fraud ring or mule network is verified, the analyst must compile the full case history including:
  - Scored transaction timeline and SHAP explanation summaries.
  - Subgraph extract of connected devices, IPs, and merchants.
  - Exported FraudLens case investigation report.
- Submit according to local statutory guidelines (e.g., FIU-IND / FinCEN / RBI guidelines) within the mandatory filing timeframe (typically 7 to 14 days from determination).
