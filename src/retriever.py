from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss

from .embedder import Embedder

@dataclass
class RetrievedChunk:
    doc_id: str
    doc_type: str
    section_id: str
    section_label: str
    page: int
    chunk_id: str
    text: str
    score: float
    
def load_chunks_jsonl(path: Path) -> List[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_faiss_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


class Retriever:
    def __init__(self, faiss_index: faiss.Index, chunks: List[dict], embedder: Embedder):
        self.index = faiss_index
        self.chunks = chunks
        self.embedder = embedder
        
    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        section_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Note: FAISS returns global nearest neighbors.
        Filtering by doc_id/doc_type/section happens AFTER retrieval.
        For large corpora, improve this with per-doc/per-type indices or prefilter lists.
        """
        q_emb = self.embedder.embed_texts([query])  # (1, d)
        scores, idxs = self.index.search(q_emb, top_k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            c = self.chunks[idx]

            if doc_id and c.get("doc_id") != doc_id:
                continue
            if doc_type and c.get("doc_type") != doc_type:
                continue
            if section_id and c.get("section_id") != section_id:
                continue

            results.append(
                RetrievedChunk(
                    doc_id=c["doc_id"],
                    doc_type=c.get("doc_type", "unknown"),
                    section_id=c.get("section_id", "UNKNOWN"),
                    section_label=c.get("section_label", c.get("section", "UNKNOWN")),
                    page=int(c["page"]),
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    score=float(score),
                )
            )

        return results
