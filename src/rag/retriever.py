"""
RAG Retriever:
Executes semantic and keyword hybrid retrieval over indexed chunks with metadata filtering and source citation extraction.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.rag.knowledge_indexer import KnowledgeIndexer, KnowledgeChunk


class RagRetriever:
    """Retrieves top-k relevant knowledge chunks with source metadata and citations."""

    def __init__(self, indexer: KnowledgeIndexer):
        self.indexer = indexer

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        doc_type: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        min_score: float = 0.03,
        deduplicate_docs: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant chunks matching query, applying optional metadata filters
        and ensuring distinct document diversity by default.
        """
        if not self.indexer.chunks or self.indexer.tfidf_matrix is None:
            self.indexer.build_index()

        if not self.indexer.chunks or self.indexer.tfidf_matrix is None:
            return []

        # Vectorize query
        q_vec = self.indexer.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.indexer.tfidf_matrix)[0]

        query_words = set(query.lower().split())

        candidates = []
        for idx, chunk in enumerate(self.indexer.chunks):
            # Apply metadata filters
            if doc_type and chunk.document_type != doc_type:
                continue
            if jurisdiction and jurisdiction.lower() not in chunk.jurisdiction.lower():
                continue

            score = float(sims[idx])

            # Boost if query tokens match title, section, or tags
            title_overlap = sum(1 for w in query_words if len(w) > 2 and w in chunk.title.lower())
            sec_overlap = sum(1 for w in query_words if len(w) > 2 and w in chunk.section_title.lower())
            tag_overlap = sum(1 for w in query_words if len(w) > 2 and any(w in t.lower() for t in chunk.tags))
            score += 0.15 * title_overlap + 0.10 * sec_overlap + 0.12 * tag_overlap

            if score >= min_score:
                candidates.append({
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "document_type": chunk.document_type,
                    "jurisdiction": chunk.jurisdiction,
                    "source": chunk.source,
                    "source_url": chunk.source_url,
                    "publication_date": chunk.publication_date,
                    "section_title": chunk.section_title,
                    "text": chunk.text,
                    "text_snippet": chunk.text[:220].replace("\n", " ").strip() + ("..." if len(chunk.text) > 220 else ""),
                    "score": round(score, 4)
                })

        candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)

        if not deduplicate_docs:
            return candidates[:top_k]

        # Deduplicate by document_id to guarantee diverse, non-repetitive knowledge sources
        seen_doc_ids = set()
        deduped = []
        for c in candidates:
            if c["document_id"] not in seen_doc_ids:
                seen_doc_ids.add(c["document_id"])
                deduped.append(c)
                if len(deduped) >= top_k:
                    break

        # Fallback if fewer distinct documents met threshold: fill with remaining candidate chunks
        if len(deduped) < top_k:
            for c in candidates:
                if c not in deduped:
                    deduped.append(c)
                    if len(deduped) >= top_k:
                        break

        return deduped

    def retrieve_context_bundle(self, query: str, signals: List[str]) -> Dict[str, Any]:
        """
        Retrieves a multi-dimensional knowledge bundle covering:
        - 1 Best-matching Fraud Typology
        - 1 Best-matching Investigation SOP
        - 1 Best-matching Regulatory Guideline (RBI/Fed)
        - 1 Best-matching Historical Benchmark Case
        """
        search_text = f"{query} {' '.join(signals)}"
        
        typology = self.retrieve(search_text, top_k=1, doc_type="fraud_typologies", min_score=0.01)
        investigation = self.retrieve(search_text, top_k=1, doc_type="investigation", min_score=0.01)
        regulatory = self.retrieve(search_text, top_k=1, doc_type="regulatory", min_score=0.01)
        historical = self.retrieve(search_text, top_k=1, doc_type="historical_cases", min_score=0.01)

        return {
            "typology": typology[0] if typology else None,
            "investigation": investigation[0] if investigation else None,
            "regulatory": regulatory[0] if regulatory else None,
            "historical": historical[0] if historical else None
        }
