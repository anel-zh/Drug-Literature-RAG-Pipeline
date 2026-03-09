from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class FusedResult:
    idx: int
    score: float
    rank_dense: int | None
    rank_bm25: int | None


def rrf_fuse(
    dense_idxs: List[int],
    bm25_idxs: List[int],
    k: int = 60,
) -> List[FusedResult]:
    """
    Reciprocal Rank Fusion on ranks (not raw scores).
    """
    rank_dense: Dict[int, int] = {idx: r for r, idx in enumerate(dense_idxs, start=1)}
    rank_bm25: Dict[int, int] = {idx: r for r, idx in enumerate(bm25_idxs, start=1)}

    all_ids = set(rank_dense) | set(rank_bm25)
    fused: List[FusedResult] = []

    for idx in all_ids:
        rd = rank_dense.get(idx)
        rb = rank_bm25.get(idx)
        score = 0.0
        if rd is not None:
            score += 1.0 / (k + rd)
        if rb is not None:
            score += 1.0 / (k + rb)

        fused.append(FusedResult(idx=idx, score=score, rank_dense=rd, rank_bm25=rb))

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused