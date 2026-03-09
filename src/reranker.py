from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

from sentence_transformers import CrossEncoder


@dataclass
class RerankResult:
    idx: int
    score: float


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[Dict], idxs: List[int], top_k: int = 8) -> List[RerankResult]:
        pairs = [(query, chunks[i]["text"]) for i in idxs]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(idxs, scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
        return [RerankResult(idx=i, score=float(s)) for i, s in ranked]