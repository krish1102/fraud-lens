---
document_id: TYP-004
title: "Fraud Typology: Mule Account & Laundering Networks"
document_type: fraud_typology
jurisdiction: Global & India (RBI Framework)
source: "Financial Crime & Regulatory Intelligence Unit"
source_url: "https://fraudlens.internal/typologies/mule-networks"
publication_date: "2024-06-18"
ingestion_date: "2024-10-01"
tags: [money_mule, entity_graph, shared_ip, pass_through, layering]
---

# Typology Overview: Money Mule Networks & Pass-Through Activity

## 1. Description
A money mule account is a deposit or payment account used by fraudsters to receive, transit, and quickly disperse proceeds of unauthorized transactions or cyber fraud. In online transaction systems, mule networks are recognized by high hub degree, shared infrastructure, and rapid outflow following inbound inflows.

## 2. Key Signals & Graph Attributes
- **Shared Entity Hub**: The same IP address (`ip_customer_count > 5`) or device fingerprint (`device_customer_count > 3`) is linked to dozens of different customer identifiers.
- **Pass-Through Velocity**: Funds deposited into the account are transferred out or utilized for gift-card / high-resale purchases within minutes (`transactions_last_30m >= 4`).
- **High Entity Degree**: High connectivity (`entity_degree >= 6`) with previously flagged suspicious neighbors (`suspicious_neighbor_count > 0`).
- **Divergent Geographic Signals**: Discrepancy between IP registration geolocation, account profile address, and delivery coordinates.

## 3. Investigation Protocol
1. Query the Entity Graph for 1st-degree and 2nd-degree neighbors of the customer, device, and IP nodes.
2. Review connected transaction cases (`connected_case_count`) to identify shared syndicate clusters.
3. Check against regulatory red-flag lists and bank-wide freeze mandates.
