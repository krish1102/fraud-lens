---
document_id: TYP-001
title: "Fraud Typology: Account Takeover (ATO)"
document_type: fraud_typology
jurisdiction: Global / E-Commerce
source: "FraudLens Knowledge Engineering Team"
source_url: "https://fraudlens.internal/typologies/account-takeover"
publication_date: "2024-03-15"
ingestion_date: "2024-10-01"
tags: [account_takeover, credential_stuffing, new_device, velocity, password_reset]
---

# Typology Overview: Account Takeover (ATO)

## 1. Description
Account Takeover (ATO) occurs when an unauthorized bad actor gains access to a legitimate customer's account credentials and conducts unauthorized transactions or profile alterations. In online transaction environments, ATO often manifests shortly after a credential stuffing attack, phishing campaign, session hijacking, or SIM swap.

## 2. Typical Behavioral Indicators & Signals
- **Device Anomaly**: Sudden transaction originating from an unrecognized hardware device (`new_device = 1`), novel browser fingerprint, or unfamiliar user-agent.
- **Network Anomaly**: Access from an unrecognized IP address (`new_ip = 1`), VPN, anonymous proxy, or hosting facility provider.
- **Velocity Spike**: High transaction velocity immediately following credential change or login from a new device (multiple transactions within 5–30 minutes).
- **Amount Deviation**: Transaction amount significantly exceeds the customer's historical average (`amount_vs_customer_avg >> 2.0x`) or standardized z-score > 3.0.
- **Shipping / Beneficiary Shift**: Sudden modification of delivery address or withdrawal to a novel beneficiary right before a high-value checkout.

## 3. Recommended Investigation Steps
1. **Verify Customer Baseline**: Compare current transaction attributes against historical customer baseline (past 90 days of normal activity).
2. **Inspect Entity Graph**: Determine if the newly introduced device ID or IP address is shared with known fraudulent accounts or suspicious clusters (`shared_device_count > 1`, `shared_ip_count > 1`).
3. **Check Temporal Sequence**: Verify whether password change, email update, or MFA modification preceded the transaction within a 24-hour window.
4. **Out-of-Band Verification**: Initiate out-of-band verification (SMS OTP / phone call confirmation) to the original phone number on file before settlement.
5. **Mitigation**: If confirmed fraudulent, revoke active sessions, freeze connected payment instruments, and alert the customer to re-authenticate.
