from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np

from .chunker import Chunk


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_chunks_jsonl(chunks: List[Chunk], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "doc_type": c.doc_type,
                        "page": c.page,
                        # NEW (canonical + display)
                        "section_id": getattr(c, "section_id", None) or "UNKNOWN",
                        "section_label": getattr(c, "section_label", None) or getattr(c, "section", "UNKNOWN"),
                        # Keep old field optional for backward compatibility if you want:
                        # "section": getattr(c, "section", None) or getattr(c, "section_label", "UNKNOWN"),
                        "text": c.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            
def save_meta(meta: Dict[str, Any], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Uses inner product index (works with normalized embeddings = cosine similarity).
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.Index, out_path: Path) -> None:
    faiss.write_index(index, str(out_path))