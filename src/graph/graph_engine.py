"""
Fraud Network & Entity Graph Engine:
Constructs a heterogeneous bipartite/multimodal NetworkX graph connecting customers,
devices, IP addresses, merchants, and beneficiaries. Extracts subgraphs and calculates graph risk metrics.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import networkx as nx
import pandas as pd
import numpy as np


class GraphEngine:
    """Heterogeneous entity graph engine for fraud investigation and shared infrastructure analysis."""

    def __init__(self):
        self.graph = nx.Graph()

    def build_from_dataframe(self, df: pd.DataFrame, flagged_customers: Optional[Set[str]] = None) -> nx.Graph:
        """
        Constructs the entity graph from canonical or enriched transaction DataFrame.
        Node types:
          - 'customer'
          - 'device'
          - 'ip'
          - 'merchant'
          - 'beneficiary'
        """
        self.graph.clear()
        flagged = flagged_customers or set()

        for _, row in df.iterrows():
            cust_id = str(row.get("customer_id", "")).strip()
            if not cust_id:
                continue

            cust_node = f"cust:{cust_id}"
            if not self.graph.has_node(cust_node):
                self.graph.add_node(
                    cust_node,
                    node_id=cust_id,
                    node_type="customer",
                    is_flagged=(cust_id in flagged),
                    label=f"Customer {cust_id}"
                )

            # Link Device
            dev_id = str(row.get("device_id", "")).strip()
            if dev_id and dev_id not in ["", "nan", "None"]:
                dev_node = f"dev:{dev_id}"
                if not self.graph.has_node(dev_node):
                    self.graph.add_node(dev_node, node_id=dev_id, node_type="device", label=f"Device {dev_id}")
                self.graph.add_edge(cust_node, dev_node, relationship="USED_DEVICE")

            # Link IP
            ip_id = str(row.get("ip_address", "")).strip()
            if ip_id and ip_id not in ["", "nan", "None"]:
                ip_node = f"ip:{ip_id}"
                if not self.graph.has_node(ip_node):
                    self.graph.add_node(ip_node, node_id=ip_id, node_type="ip", label=f"IP {ip_id}")
                self.graph.add_edge(cust_node, ip_node, relationship="LOGGED_FROM_IP")

            # Link Merchant
            merch_id = str(row.get("merchant_id", "")).strip()
            if merch_id and merch_id not in ["", "nan", "None"]:
                merch_node = f"merch:{merch_id}"
                if not self.graph.has_node(merch_node):
                    self.graph.add_node(merch_node, node_id=merch_id, node_type="merchant", label=f"Merchant {merch_id}")
                self.graph.add_edge(cust_node, merch_node, relationship="TRANSACTED_WITH")

            # Link Beneficiary if present
            ben_id = str(row.get("beneficiary_id", "")).strip()
            if ben_id and ben_id not in ["", "nan", "None"]:
                ben_node = f"ben:{ben_id}"
                if not self.graph.has_node(ben_node):
                    self.graph.add_node(ben_node, node_id=ben_id, node_type="beneficiary", label=f"Beneficiary {ben_id}")
                self.graph.add_edge(cust_node, ben_node, relationship="TRANSFERRED_TO")

        return self.graph

    def get_customer_network_metrics(self, customer_id: str) -> Dict[str, Any]:
        """
        Calculates graph structural features and evidence for a specific customer.
        """
        cust_node = f"cust:{customer_id}"
        if not self.graph.has_node(cust_node):
            return {
                "entity_degree": 0,
                "shared_device_count": 0,
                "shared_ip_count": 0,
                "device_customer_count": 1,
                "ip_customer_count": 1,
                "suspicious_neighbor_count": 0,
                "network_risk": 0.0,
                "summary": "Customer has no graph connections recorded."
            }

        neighbors = list(self.graph.neighbors(cust_node))
        entity_degree = len(neighbors)

        connected_customers: Set[str] = set()
        shared_devices: List[str] = []
        shared_ips: List[str] = []
        suspicious_neighbors = 0

        for neighbor in neighbors:
            ntype = self.graph.nodes[neighbor].get("node_type")
            # Look at 2nd hop customers through shared entities
            if ntype in ["device", "ip"]:
                second_hops = [
                    n for n in self.graph.neighbors(neighbor)
                    if n.startswith("cust:") and n != cust_node
                ]
                if second_hops:
                    connected_customers.update(second_hops)
                    if ntype == "device":
                        shared_devices.append(self.graph.nodes[neighbor].get("node_id"))
                    elif ntype == "ip":
                        shared_ips.append(self.graph.nodes[neighbor].get("node_id"))

                # Check if any second hop customer was flagged
                for sh in second_hops:
                    if self.graph.nodes[sh].get("is_flagged", False):
                        suspicious_neighbors += 1

        # Heuristic Network Risk Score [0.0, 1.0]
        # Base risk proportional to shared devices, shared IPs, and suspicious neighbors
        raw_risk = (
            0.35 * min(1.0, len(shared_devices) / 2.0) +
            0.25 * min(1.0, len(shared_ips) / 3.0) +
            0.30 * min(1.0, suspicious_neighbors / 2.0) +
            0.10 * min(1.0, len(connected_customers) / 5.0)
        )
        network_risk = round(float(np.clip(raw_risk, 0.0, 1.0)), 2)

        summary_parts = []
        if shared_devices:
            summary_parts.append(f"Shares {len(shared_devices)} device(s) with {len(connected_customers)} other customer(s)")
        if shared_ips:
            summary_parts.append(f"Shares IP infrastructure with multiple entities")
        if suspicious_neighbors > 0:
            summary_parts.append(f"Directly linked to {suspicious_neighbors} previously flagged suspicious entity/cases")
        if not summary_parts:
            summary_parts.append("Isolated entity; no abnormal cross-customer sharing detected")

        return {
            "entity_degree": entity_degree,
            "shared_device_count": len(shared_devices),
            "shared_ip_count": len(shared_ips),
            "connected_customers_count": len(connected_customers),
            "suspicious_neighbor_count": suspicious_neighbors,
            "network_risk": network_risk,
            "summary": "; ".join(summary_parts)
        }

    def get_subgraph_data(self, customer_id: str, max_hops: int = 2) -> Dict[str, Any]:
        """
        Extracts an ego network around a customer for visualization.
        Returns JSON-serializable nodes and edges lists.
        """
        cust_node = f"cust:{customer_id}"
        if not self.graph.has_node(cust_node):
            return {"nodes": [], "edges": []}

        # BFS to extract nodes up to max_hops
        sub_nodes = set()
        current_layer = {cust_node}
        sub_nodes.update(current_layer)

        for _ in range(max_hops):
            next_layer = set()
            for node in current_layer:
                next_layer.update(self.graph.neighbors(node))
            next_layer.difference_update(sub_nodes)
            sub_nodes.update(next_layer)
            current_layer = next_layer
            if len(sub_nodes) > 50:  # Cap visualization size for performance
                break

        subgraph = self.graph.subgraph(sub_nodes)

        nodes = []
        for n in subgraph.nodes():
            n_data = subgraph.nodes[n]
            nodes.append({
                "id": str(n),
                "node_id": str(n_data.get("node_id", n)),
                "label": str(n_data.get("label", n)),
                "type": str(n_data.get("node_type", "unknown")),
                "is_flagged": bool(n_data.get("is_flagged", False)),
                "degree": int(subgraph.degree[n])
            })

        edges = []
        for u, v, d in subgraph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "relationship": d.get("relationship", "CONNECTED")
            })

        return {"nodes": nodes, "edges": edges}
