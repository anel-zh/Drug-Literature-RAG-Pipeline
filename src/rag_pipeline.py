from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import time

from src.retriever import Retriever
from src.bm25_store import BM25Store
from src.fusion import rrf_fuse
from src.reranker import CrossEncoderReranker
from src.llm_local import LocalLLM
from src.router_rules import route_query, RoutingDecision


@dataclass
class EvidenceChunk:
    doc_id: str
    page: int
    text: str


class AdvancedHybridRAG:
    """
    Production-grade advanced pipeline:
    Router (metadata filter) + Dense + BM25 + RRF + Cross-Encoder rerank
    with starvation protection.
    """

    def __init__(
        self,
        chunks: List[dict],
        retriever: Retriever,
        bm25: BM25Store,
        reranker: CrossEncoderReranker,
        llm: LocalLLM,
        meta: dict,
        *,
        dense_k: int = 60,
        bm25_k: int = 60,
        fused_candidates: int = 60,
        final_k: int = 12,
        max_evid_blocks: int = 6,
        max_evid_chars: int = 1400,
        subset_fallback_k: int = 10,
    ):
        self.chunks = chunks
        self.retriever = retriever
        self.bm25 = bm25
        self.reranker = reranker
        self.llm = llm
        self.meta = meta

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.fused_candidates = fused_candidates
        self.final_k = final_k

        self.max_evid_blocks = max_evid_blocks
        self.max_evid_chars = max_evid_chars
        self.subset_fallback_k = subset_fallback_k

        self.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        
    def _allowed_idxs(self, decision: RoutingDecision) -> set[int]:
        allowed = set()
        for i, c in enumerate(self.chunks):
            if decision.doc_ids and c.get("doc_id") not in decision.doc_ids:
                continue
            if decision.doc_type and c.get("doc_type") != decision.doc_type:
                continue
            if decision.section_ids and c.get("section_id") not in decision.section_ids:
                continue
            allowed.add(i)
        return allowed

    def retrieve_ranked_indices(self, query: str) -> Tuple[List[int], RoutingDecision]:
        decision = route_query(query, self.meta)
        allowed_idxs = self._allowed_idxs(decision)

        # 1) Global searches (keep semantic scores)
        dense = self.retriever.search(query, top_k=self.dense_k)
        bm25_res = self.bm25.search(query, top_k=self.bm25_k)

        # 2) Extract indices
        raw_dense_idxs = [
            self.chunk_id_to_idx[r.chunk_id]
            for r in dense
            if r.chunk_id in self.chunk_id_to_idx
        ]
        raw_bm25_idxs = [r.idx for r in bm25_res]

        # 3) Apply router filter AFTER retrieval
        if allowed_idxs:
            dense_idxs = [i for i in raw_dense_idxs if i in allowed_idxs]
            bm25_idxs = [i for i in raw_bm25_idxs if i in allowed_idxs]

            # CRITICAL: starvation protection (identical to your Script 05)
            if len(dense_idxs) + len(bm25_idxs) == 0:
                # Force-include chunks from allowed section by subset BM25
                bm25_idxs = self.bm25.search_in_subset(query, list(allowed_idxs), top_k=self.subset_fallback_k)
                dense_idxs = []
        else:
            dense_idxs = raw_dense_idxs
            bm25_idxs = raw_bm25_idxs

        # 4) Fuse + rerank
        fused = rrf_fuse(dense_idxs, bm25_idxs, k=60)[: self.fused_candidates]
        fused_idxs = [f.idx for f in fused]
        reranked = self.reranker.rerank(query, self.chunks, fused_idxs, top_k=self.final_k)

        ranked_idxs = [r.idx for r in reranked]
        return ranked_idxs, decision

    def build_evidence(self, query: str) -> Tuple[str, RoutingDecision]:
        ranked_idxs, decision = self.retrieve_ranked_indices(query)

        blocks = []
        for idx in ranked_idxs[: self.max_evid_blocks]:
            c = self.chunks[idx]
            text = (c.get("text") or "")[: self.max_evid_chars]
            sec = c.get("section_label") or c.get("section_id") or "UNKNOWN"
            blocks.append(f"[{c['doc_id']} p.{c['page']} — {sec}]\n{text}\n")

        return "\n---\n".join(blocks), decision

    def generate_answer(
        self,
        query: str,
        system_prompt: str,
        *,
        answer_max_tokens: int,
        llm_num_ctx: int,
        stream: bool = False,
    ):
        evidence_text, decision = self.build_evidence(query)
        user_prompt = f"""Question:
{query}

Evidence excerpts:
{evidence_text}
"""

        if not stream:
            ans = self.llm.generate(
                system_prompt,
                user_prompt,
                options={
                    "temperature": 0.1,
                    "num_predict": answer_max_tokens,
                    "top_p": 0.9,
                    "num_ctx": llm_num_ctx,
                },
            )
            return ans, decision

        # streaming generator
        gen = self.llm.stream_lines(
            system_prompt,
            user_prompt,
            options={
                "temperature": 0.1,
                "num_predict": answer_max_tokens,
                "top_p": 0.9,
                "num_ctx": llm_num_ctx,
            },
        )
        return gen, decision


class VanillaDenseRAG:
    """
    Baseline pipeline:
    Dense only, no router, no BM25, no fusion, no rerank.
    Used for benchmarking.
    """

    def __init__(
        self,
        chunks: List[dict],
        retriever: Retriever,
        llm: LocalLLM,
        *,
        dense_k: int = 5,
        max_evid_chars: int = 1400,
    ):
        self.chunks = chunks
        self.retriever = retriever
        self.llm = llm
        self.dense_k = dense_k
        self.max_evid_chars = max_evid_chars
        self.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    def build_evidence(self, query: str) -> str:
        dense = self.retriever.search(query, top_k=self.dense_k)
        blocks = []
        for r in dense:
            idx = self.chunk_id_to_idx.get(r.chunk_id)
            if idx is None:
                continue
            c = self.chunks[idx]
            text = (c.get("text") or "")[: self.max_evid_chars]
            blocks.append(text)
        return "\n".join(blocks)

    def generate_answer(
        self,
        query: str,
        prompt: str,
        *,
        answer_max_tokens: int,
        llm_num_ctx: int,
    ) -> str:
        evidence = self.build_evidence(query)
        user = f"Context:\n{evidence}\n\nQuestion: {query}"
        return self.llm.generate(
            prompt,
            user,
            options={
                "temperature": 0.2,
                "num_predict": answer_max_tokens,
                "top_p": 0.9,
                "num_ctx": llm_num_ctx,
            },
        )