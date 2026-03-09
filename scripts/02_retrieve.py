from __future__ import annotations

import json
import time
from pathlib import Path

from src.config import Settings
from src.embedder import Embedder
from src.retriever import load_chunks_jsonl, load_faiss_index, Retriever
from src.bm25_store import BM25Store
from src.reranker import CrossEncoderReranker
from src.llm_local import LocalLLM
from src.rag_pipeline import AdvancedHybridRAG
from src.prompts import ADVANCED_SYSTEM_PROMPT


def load_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main():
    s = Settings()

    chunks = load_chunks_jsonl(Path(s.chunks_path))
    index = load_faiss_index(Path(s.faiss_index_path))
    meta = load_meta(Path(s.meta_path))

    embedder = Embedder(s.embedding_model)
    retriever = Retriever(index, chunks, embedder)
    bm25 = BM25Store(chunks)
    reranker = CrossEncoderReranker()
    llm = LocalLLM(model=s.local_llm_model)

    pipeline = AdvancedHybridRAG(
        chunks=chunks,
        retriever=retriever,
        bm25=bm25,
        reranker=reranker,
        llm=llm,
        meta=meta,
    )

    print("\nHybrid RAG (FAISS + BM25 + RRF + CrossEncoder) with citations. Type 'exit' to quit.\n")

    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        t0 = time.time()

        gen, decision = pipeline.generate_answer(
            q,
            ADVANCED_SYSTEM_PROMPT,
            answer_max_tokens=s.answer_max_tokens,
            llm_num_ctx=s.llm_num_ctx,
            stream=True,
        )

        print(
            f"\n[ROUTER] doc_ids={decision.doc_ids or 'ALL'} | doc_type={decision.doc_type or 'ALL'} | sections={decision.sections or 'ALL'}"
        )
        print("\n--- Answer (streaming) ---\n")

        for line in gen:
            print(line, end="", flush=True)

        dt = time.time() - t0
        print(f"\n--- End ---\n[INFO] latency={dt:.2f}s\n")


if __name__ == "__main__":
    main()