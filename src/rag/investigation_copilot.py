"""
Investigation Copilot:
Answers analyst questions regarding flagged cases, typologies, SOPs, and RBI regulatory guidance.
Enforces strict grounding, source citations, and separation of ML evidence from policy interpretation.
"""

from typing import Dict, Any, List, Optional
import os
import re
from src.rag.retriever import RagRetriever


class InvestigationCopilot:
    """RAG-powered investigative copilot for fraud case triage and regulatory grounding."""

    def __init__(self, retriever: RagRetriever):
        self.retriever = retriever

    def answer_query(
        self,
        query: str,
        case_data: Dict[str, Any],
        jurisdiction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes query in the context of an investigation case and retrieved knowledge.
        Returns:
            - 'answer': formatted, grounded response with citations
            - 'sources': list of source citations used in the response
            - 'intent': detected query intent
            - 'case_id': associated case identifier
        """
        query_lower = query.lower()

        # Intent detection
        intent = "general_investigation"
        doc_type_filter = None

        if any(w in query_lower for w in ["why", "flagged", "reason", "score", "priority", "why risk", "high risk"]):
            intent = "why_flagged"
        elif any(w in query_lower for w in ["typology", "pattern", "resemble", "kind of fraud", "scheme", "attack", "type"]):
            intent = "fraud_typology"
            doc_type_filter = "fraud_typologies"
        elif any(w in query_lower for w in ["next", "step", "check", "verify", "sop", "procedure", "what to do", "action"]):
            intent = "investigation_steps"
            doc_type_filter = "investigation"
        elif any(w in query_lower for w in ["rbi", "regulatory", "direction", "compliance", "law", "ews", "early warning", "central fraud"]):
            intent = "regulatory_guidance"
            doc_type_filter = "regulatory"
        elif any(w in query_lower for w in ["similar", "historical", "benchmark", "past cases", "case comparison"]):
            intent = "similar_cases"
            doc_type_filter = "historical_cases"

        signals = case_data.get("detected_signals", [])
        if isinstance(signals, str):
            try:
                import json
                signals = json.loads(signals)
            except Exception:
                signals = [signals]

        signals_text = " ".join(signals)
        retrieval_query = f"{query} {signals_text}"

        chunks = []
        context_bundle = {}

        if intent == "why_flagged":
            # Multi-dimensional grounding: pull best typology, regulatory guidance, and SOP
            context_bundle = self.retriever.retrieve_context_bundle(query, signals)
            for k in ["typology", "regulatory", "investigation"]:
                if context_bundle.get(k):
                    chunks.append(context_bundle[k])
        else:
            chunks = self.retriever.retrieve(
                retrieval_query,
                top_k=3,
                doc_type=doc_type_filter,
                jurisdiction=jurisdiction,
                deduplicate_docs=True
            )
            # Fallback if filtered retrieval returned nothing
            if not chunks:
                chunks = self.retriever.retrieve(retrieval_query, top_k=3, deduplicate_docs=True)

        # Build distinct citations list
        citations = []
        seen_docs = set()
        for c in chunks:
            if c["document_id"] not in seen_docs:
                seen_docs.add(c["document_id"])
                citations.append({
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "title": c["title"],
                    "section_title": c.get("section_title", "General"),
                    "document_type": c["document_type"],
                    "jurisdiction": c["jurisdiction"],
                    "source": c["source"],
                    "source_url": c["source_url"],
                    "publication_date": c["publication_date"],
                    "relevance_score": c.get("score", 0.85)
                })

        # Synthesize deeply grounded response
        answer = self._synthesize_response(query, intent, case_data, signals, chunks, context_bundle, citations)

        return {
            "case_id": case_data.get("case_id", "N/A"),
            "query": query,
            "intent": intent,
            "answer": answer,
            "sources": citations
        }

    def _synthesize_response(
        self,
        query: str,
        intent: str,
        case: Dict[str, Any],
        signals: List[str],
        chunks: List[Dict[str, Any]],
        bundle: Dict[str, Any],
        citations: List[Dict[str, Any]]
    ) -> str:
        """Constructs an evidence-backed, fully grounded response aligned with project plan specs."""
        case_id = case.get("case_id", "Unassigned")
        txn_id = case.get("transaction_id", "N/A")
        cust_id = case.get("customer_id", "N/A")
        fraud_p = case.get("fraud_probability", 0.0)
        anom_s = case.get("anomaly_score", 0.0)
        net_r = case.get("network_risk", 0.0)
        comb_s = case.get("combined_score", 0.0)
        risk_band = case.get("risk_band", "HIGH")
        shap_summary = case.get("shap_summary", "")

        sections = []

        # =========================================================================
        # 1. INTENT: WHY FLAGGED (Comprehensive Triage & Grounded Synthesis)
        # =========================================================================
        if intent == "why_flagged":
            sections.append(f"### 🛡️ Investigation Triage & Flagging Rationale — Case #{case_id}")
            sections.append(
                f"Transaction **`{txn_id}`** for analytical customer **`{cust_id}`** was flagged as **{risk_band} RISK** "
                f"(Combined Priority Score: **`{comb_s:.2f}`** / 1.00). The alert was triggered by a confluence of "
                f"statistical behavioral deviation, machine learning pattern matching, and entity network risk."
            )

            # Plain English Risk Driver Breakdown
            sections.append("\n#### 1. Core Risk Factors & Telemetry")
            if fraud_p >= 0.70:
                sections.append(f"- **Known Fraud Pattern Alignment (`{fraud_p:.2f}`)**: The supervised LightGBM model detected high interaction similarity with verified fraud vectors.")
            if anom_s >= 0.70:
                sections.append(f"- **Unsupervised Baseline Outlier (`{anom_s:.2f}`)**: The Isolation Forest flagged significant divergence from the customer's typical expenditure velocity.")
            if net_r >= 0.40:
                sections.append(f"- **Entity Network Collision (`{net_r:.2f}`)**: The graph engine identified shared hardware or network infrastructure with multiple distinct accounts.")

            if signals:
                sections.append("\n**Observed Behavioral Signals:**")
                for s in signals[:6]:
                    sections.append(f"- 🔴 {s}")

            # Grounded Fraud Typology Alignment
            typology = bundle.get("typology") or (chunks[0] if chunks else None)
            if typology:
                sections.append("\n#### 2. Matched Fraud Typology Grounding")
                sections.append(
                    f"This activity pattern closely correlates with **{typology['title']}** `[{typology['document_id']}]` "
                    f"*(Source: {typology['source']})*:\n"
                    f"> *\"{typology['text_snippet']}\"*\n\n"
                    f"**Why this applies**: When a transaction combines an abrupt amount surge with an unrecognized device fingerprint "
                    f"or shared IP cluster, it indicates either an **Account Takeover (ATO)** attempt or an unauthorized syndicate operation."
                )

            # Grounded Regulatory Directives
            regulatory = bundle.get("regulatory")
            if regulatory:
                sections.append("\n#### 3. Regulatory Standing (RBI Master Directions)")
                sections.append(
                    f"Under **{regulatory['title']}** `[{regulatory['document_id']}]`:\n"
                    f"- This sudden deviation constitutes an **Early Warning Signal (EWS)** requiring proactive escalation before financial losses settle.\n"
                    f"- **Natural Justice Clause**: RBI directions state that automated algorithmic scores represent investigatory alerts and do not "
                    f"alone constitute legal proof of fraud. Human analyst review and customer confirmation are required prior to adverse reporting."
                )

            # Immediate SOP Next Steps
            sop = bundle.get("investigation")
            if sop:
                sections.append("\n#### 4. Immediate Analyst Action Checklist")
                sections.append(
                    f"In accordance with **{sop['title']}** `[{sop['document_id']}]`:\n"
                    f"1. **Hold Settlement**: Place a temporary settlement hold on transaction `{txn_id}`.\n"
                    f"2. **Out-of-Band Verification**: Challenge customer `{cust_id}` via registered mobile phone OTP or verified contact channel.\n"
                    f"3. **Graph Inspection**: Review Step 6 (Network Graph) to determine if sibling customer accounts linked to this device should be placed under watch."
                )

        # =========================================================================
        # 2. INTENT: FRAUD TYPOLOGY (Pattern Matching & Typology Breakdown)
        # =========================================================================
        elif intent == "fraud_typology":
            sections.append(f"### 🚨 Fraud Typology & Attack Pattern Analysis — Case #{case_id}")
            if chunks:
                top_typ = chunks[0]
                sections.append(
                    f"Based on behavioral signals (`{', '.join(signals[:3]) if signals else 'amount & velocity deviation'}`), "
                    f"this case most closely aligns with:\n\n"
                    f"### **{top_typ['title']}** `[{top_typ['document_id']}]`\n"
                    f"*Category: {top_typ['document_type']} | Jurisdiction: {top_typ['jurisdiction']} | Source: {top_typ['source']}*\n\n"
                    f"{top_typ['text']}\n"
                )
                sections.append("\n#### Signal Correlation Matrix:")
                sections.append("| Typology Indicator | Observed in Active Case | Analytical Assessment |")
                sections.append("| :--- | :--- | :--- |")
                for s in signals[:5]:
                    sections.append(f"| Recognized Behavioral Marker | `{s}` | ⚠️ Matched Signal |")
            else:
                sections.append("No specific fraud typology document reached the confidence threshold for this transaction.")

        # =========================================================================
        # 3. INTENT: INVESTIGATION STEPS (Analyst SOP & Verification Checklist)
        # =========================================================================
        elif intent == "investigation_steps":
            sections.append(f"### 📋 Standard Operating Procedure (SOP) Checklist — Case #{case_id}")
            sop_chunk = chunks[0] if chunks else None
            if sop_chunk:
                sections.append(
                    f"Grounded in **{sop_chunk['title']}** `[{sop_chunk['document_id']}]` "
                    f"*(Source: {sop_chunk['source']})*:\n\n"
                    f"{sop_chunk['text']}\n"
                )
            else:
                sections.append(
                    "#### Standard FraudLens Operational Protocol:\n"
                    "1. **Baseline Cross-Examination**: Compare current transaction amount against the customer's 90-day baseline median.\n"
                    "2. **Device Hardware Verification**: Check whether the device fingerprint has been observed on other accounts in the entity graph.\n"
                    "3. **Customer Identity Challenge**: Initiate out-of-band challenge (SMS OTP or phone callback) to customer phone on file.\n"
                    "4. **Temporary Merchant Hold**: Delay payment capture until user confirmation is logged.\n"
                    "5. **Audit Registry Update**: Record disposition notes and export formal investigation report."
                )

        # =========================================================================
        # 4. INTENT: REGULATORY GUIDANCE (RBI Directives & Compliance Requirements)
        # =========================================================================
        elif intent == "regulatory_guidance":
            sections.append(f"### 🏛️ Regulatory Directives & RBI Compliance Guidance")
            reg_chunk = chunks[0] if chunks else None
            if reg_chunk:
                sections.append(
                    f"Reference: **{reg_chunk['title']}** `[{reg_chunk['document_id']}]`\n"
                    f"*Jurisdiction: {reg_chunk['jurisdiction']} | Official Source: {reg_chunk['source']}*\n\n"
                    f"{reg_chunk['text']}\n\n"
                    f"#### 📌 Mandatory Compliance Summary for Case #{case_id}:\n"
                    f"- **Early Warning Signals (EWS)**: Rapid velocity spikes and hardware collisions constitute mandatory alert indicators under RBI Master Directions.\n"
                    f"- **Central Fraud Registry (CFR)**: If confirmed as an organized syndicate or unauthorized access, reporting must occur within the statutory window.\n"
                    f"- **Natural Justice**: Machine learning probabilities provide statistical guidance, not legal proof. The customer must be granted an opportunity to verify unusual activity before account freezing."
                )
            else:
                sections.append("Official regulatory documentation matching this inquiry was not located in the repository.")

        # =========================================================================
        # 5. INTENT: SIMILAR BENCHMARK CASES (Historical Comparison)
        # =========================================================================
        elif intent == "similar_cases":
            sections.append(f"### 📁 Synthetic Historical Benchmark Case Comparison")
            sections.append(
                "> **Compliance Notice**: *Benchmark cases in FraudLens are synthetic analytical test cases generated "
                "for research and validation. They do not represent real individuals or live banking PII.*\n"
            )
            hist_chunks = [c for c in chunks if c.get("document_type") == "historical_cases"] or chunks
            if hist_chunks:
                for h in hist_chunks[:2]:
                    sections.append(
                        f"#### Benchmark Reference: **{h['title']}** `[{h['document_id']}]`\n"
                        f"*Source: {h['source']} ({h['publication_date']})*\n\n"
                        f"{h['text']}\n"
                    )
            else:
                sections.append("No benchmark historical case matched the query criteria.")

        # =========================================================================
        # 6. GENERAL QUERY SYNTHESIS
        # =========================================================================
        else:
            sections.append(f"### 🔍 Case Investigation Intelligence — #{case_id}")
            sections.append(f"**Inquiry**: *\"{query}\"*\n")
            if chunks:
                for c in chunks[:2]:
                    sections.append(
                        f"#### **{c['title']}** — *{c.get('section_title', 'Overview')}* `[{c['document_id']}]`\n"
                        f"*Category: {c['document_type']} | Jurisdiction: {c['jurisdiction']} | Source: {c['source']}*\n\n"
                        f"{c['text']}\n"
                    )
            else:
                sections.append("No relevant knowledge base sections directly matched this query.")

        return "\n".join(sections)
