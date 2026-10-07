"""
RAG Knowledge Indexer:
Parses markdown files in knowledge_base/, extracts YAML metadata,
splits into structured chunks with source citations, and creates index.
"""

from typing import List, Dict, Any, Optional
import os
import re
from pathlib import Path
import json
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from src.configs.settings import KNOWLEDGE_BASE_DIR, DATA_DIR


class KnowledgeChunk:
    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        title: str,
        document_type: str,
        jurisdiction: str,
        source: str,
        source_url: str,
        publication_date: str,
        section_title: str,
        text: str,
        tags: List[str]
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.title = title
        self.document_type = document_type
        self.jurisdiction = jurisdiction
        self.source = source
        self.source_url = source_url
        self.publication_date = publication_date
        self.section_title = section_title
        self.text = text
        self.tags = tags

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "document_type": self.document_type,
            "jurisdiction": self.jurisdiction,
            "source": self.source,
            "source_url": self.source_url,
            "publication_date": self.publication_date,
            "section_title": self.section_title,
            "text": self.text,
            "tags": self.tags
        }


class KnowledgeIndexer:
    """Indexes markdown knowledge base documents and prepares vector search representations."""

    def __init__(self, kb_dir: Optional[Path] = None):
        self.kb_dir = kb_dir or KNOWLEDGE_BASE_DIR
        self.chunks: List[KnowledgeChunk] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
        self.tfidf_matrix = None
        self.index_path = DATA_DIR / "rag_index.json"

    def parse_markdown_file(self, filepath: Path) -> List[KnowledgeChunk]:
        """Reads a markdown file with optional YAML frontmatter and splits into section chunks."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Use parent directory name as canonical document_type (matches UI filter values)
        metadata = {
            "document_id": filepath.stem,
            "title": filepath.stem.replace("_", " ").title(),
            "document_type": filepath.parent.name,  # e.g. fraud_typologies, regulatory, investigation, historical_cases
            "jurisdiction": "Global",
            "source": "Internal Policy",
            "source_url": "",
            "publication_date": "2024-01-01",
            "tags": []
        }

        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    yaml_meta = yaml.safe_load(parts[1])
                    if isinstance(yaml_meta, dict):
                        # Preserve directory-based document_type; let YAML override everything else
                        dir_doc_type = metadata["document_type"]
                        metadata.update(yaml_meta)
                        metadata["document_type"] = dir_doc_type  # always use folder name
                except Exception:
                    pass
                body = parts[2].strip()

        # Split body into sections by markdown headers (## or #)
        sections = re.split(r"(?m)^(?=#{1,3}\s+)", body)
        chunks = []

        chunk_counter = 1
        for sec in sections:
            sec_text = sec.strip()
            if not sec_text or len(sec_text) < 30:
                continue

            # Extract section title if present
            first_line = sec_text.split("\n", 1)[0].strip()
            section_title = first_line.lstrip("#").strip() if first_line.startswith("#") else "General Overview"

            chunk_id = f"{metadata['document_id']}-CHK{str(chunk_counter).zfill(2)}"
            chunk = KnowledgeChunk(
                chunk_id=chunk_id,
                document_id=str(metadata.get("document_id")),
                title=str(metadata.get("title")),
                document_type=str(metadata.get("document_type")),
                jurisdiction=str(metadata.get("jurisdiction")),
                source=str(metadata.get("source")),
                source_url=str(metadata.get("source_url")),
                publication_date=str(metadata.get("publication_date")),
                section_title=section_title,
                text=sec_text,
                tags=metadata.get("tags", [])
            )
            chunks.append(chunk)
            chunk_counter += 1

        return chunks

    def build_index(self) -> int:
        """Finds all markdown files across knowledge_base subdirectories and builds index."""
        self.chunks.clear()
        md_files = list(self.kb_dir.rglob("*.md"))

        for fpath in md_files:
            file_chunks = self.parse_markdown_file(fpath)
            self.chunks.extend(file_chunks)

        if not self.chunks:
            return 0

        # Build TF-IDF search space
        corpus = [f"{c.title} {c.section_title} {' '.join(c.tags)} {c.text}" for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # Save to disk
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, indent=2)

        return len(self.chunks)

    def load_index(self) -> bool:
        """Loads index from disk if present."""
        if not self.index_path.exists():
            return False
        with open(self.index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = [KnowledgeChunk(**item) for item in data]
        if self.chunks:
            corpus = [f"{c.title} {c.section_title} {' '.join(c.tags)} {c.text}" for c in self.chunks]
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            return True
        return False

    def get_document_catalog(self, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Groups indexed chunks by canonical document_id for structured browsing."""
        docs = {}
        for c in self.chunks:
            if filter_type and filter_type != "All" and c.document_type != filter_type:
                continue
            if c.document_id not in docs:
                docs[c.document_id] = {
                    "document_id": c.document_id,
                    "title": c.title,
                    "document_type": c.document_type,
                    "jurisdiction": c.jurisdiction,
                    "source": c.source,
                    "source_url": c.source_url,
                    "publication_date": c.publication_date,
                    "tags": c.tags,
                    "sections": []
                }
            docs[c.document_id]["sections"].append({
                "chunk_id": c.chunk_id,
                "section_title": c.section_title,
                "text": c.text
            })
        return list(docs.values())
