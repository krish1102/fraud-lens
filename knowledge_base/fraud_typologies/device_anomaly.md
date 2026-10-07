---
document_id: TYP-005
title: "Fraud Typology: Device Spoofing & Farm Anomaly"
document_type: fraud_typology
jurisdiction: Global / E-Commerce
source: "Cyber Threat Intelligence Group"
source_url: "https://fraudlens.internal/typologies/device-anomaly"
publication_date: "2024-07-05"
ingestion_date: "2024-10-01"
tags: [device_anomaly, device_spoofing, emulator, device_farm, hardware_fingerprint]
---

# Typology Overview: Device Spoofing & Device Farm Anomalies

## 1. Description
Device spoofing and device farm operations involve organized syndicates running emulators, automated scripting frameworks, or physical racks of mobile handsets to systematically manipulate e-commerce promotions, test stolen cards, or execute account takeovers.

## 2. Key Signals & Metrics
- **Novel Device Flag**: `new_device = 1` for an established customer profile that previously possessed a consistent device history.
- **Extreme Device Multi-Tenancy**: The device entity connects to multiple customer IDs (`device_customer_count >= 4`), indicating an emulator or device farm.
- **Rapid Switching**: Multiple customer IDs logging in from the same device within a short time window.
- **Unusual Operating Parameters**: Missing browser user-agents, impossible hardware clock offsets, or headless automation signatures.

## 3. Recommended Investigation Steps
1. Inspect the entity graph for other customer accounts associated with this `device_id`.
2. Evaluate SHAP feature contribution for `new_device` and `shared_device_count`.
3. Check if recent transactions on the same device resulted in chargebacks or fraud alerts.
4. Enforce hardware-backed multi-factor authentication (FIDO2 / WebAuthn / biometric prompt).
