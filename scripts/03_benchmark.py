from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any

from src.config import Settings
from src.embedder import Embedder
from src.retriever import load_chunks_jsonl, load_faiss_index, Retriever
from src.bm25_store import BM25Store
from src.reranker import CrossEncoderReranker
from src.llm_local import LocalLLM
from src.rag_pipeline import AdvancedHybridRAG, VanillaDenseRAG
from src.prompts import VANILLA_PROMPT, ADVANCED_SYSTEM_PROMPT, JUDGE_PROMPT


CIT_RX = re.compile(r"\[[A-Z0-9_]+ p\.\d+\]")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def safe_parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                return {"winner": "Tie", "reason": "Invalid JSON"}
        return {"winner": "Tie", "reason": "Parsing failed"}


def score_answer(answer: str) -> Dict[str, Any]:
    return {
        "has_citation": bool(CIT_RX.search(answer)),
        "is_empty": (not answer.strip()) or ("information incomplete" in answer.lower()) or ("couldn't find" in answer.lower()),
        "length": len(answer.strip()),
    }


def main():
    s = Settings()

    chunks = load_chunks_jsonl(Path(s.chunks_path))
    index = load_faiss_index(Path(s.faiss_index_path))
    meta = load_meta(Path(s.meta_path))
    questions = load_jsonl(Path(s.eval_questions_path))

    embedder = Embedder(s.embedding_model)
    retriever = Retriever(index, chunks, embedder)
    bm25 = BM25Store(chunks)
    reranker = CrossEncoderReranker()
    llm = LocalLLM(model=s.local_llm_model)

    vanilla = VanillaDenseRAG(chunks=chunks, retriever=retriever, llm=llm)
    advanced = AdvancedHybridRAG(
        chunks=chunks,
        retriever=retriever,
        bm25=bm25,
        reranker=reranker,
        llm=llm,
        meta=meta,
    )

    results = []
    stats = {"Advanced": 0, "Vanilla": 0, "Tie": 0}

    print(f"\n Benchmarking ({len(questions)} questions)")
    print("Running: Vanilla (dense-only) vs Advanced (hybrid+routing+rerank)\n")

    for q in questions:
        query = q["query"]
        qid = q.get("id", query[:30])
        print(f"Q: {qid}")

        # 1) Vanilla
        t0 = time.time()
        ans_v = vanilla.generate_answer(
            query,
            VANILLA_PROMPT,
            answer_max_tokens=min(220, s.answer_max_tokens),
            llm_num_ctx=s.llm_num_ctx,
        )
        lat_v = time.time() - t0

        # 2) Advanced
        t0 = time.time()
        ans_a, decision = advanced.generate_answer(
            query,
            ADVANCED_SYSTEM_PROMPT,
            answer_max_tokens=s.answer_max_tokens,
            llm_num_ctx=s.llm_num_ctx,
            stream=False,
        )
        lat_a = time.time() - t0

        # 3) Deterministic scoring (primary)
        sv = score_answer(ans_v)
        sa = score_answer(ans_a)

        # Determine winner deterministically:
        # - Prefer non-empty answers
        # - Prefer citation presence
        # - Prefer longer (only if both non-empty)
        winner = "Tie"
        if sv["is_empty"] and not sa["is_empty"]:
            winner = "Advanced"
        elif sa["is_empty"] and not sv["is_empty"]:
            winner = "Vanilla"
        else:
            if sa["has_citation"] and not sv["has_citation"]:
                winner = "Advanced"
            elif sv["has_citation"] and not sa["has_citation"]:
                winner = "Vanilla"
            else:
                # both same citation status -> choose longer by small margin
                if sa["length"] > sv["length"] + 30:
                    winner = "Advanced"
                elif sv["length"] > sa["length"] + 30:
                    winner = "Vanilla"
                else:
                    winner = "Tie"

        # 4) Optional judge (secondary, informational only)
        judge_raw = llm.generate(
            "You are a neutral judge.",
            JUDGE_PROMPT.format(query=query, answer_v=ans_v, answer_a=ans_a),
            options={"temperature": 0.0, "num_predict": 200, "top_p": 0.9, "num_ctx": 2048},
        )
        judge = safe_parse_json(judge_raw)

        stats[winner] += 1

        results.append(
            {
                "id": qid,
                "query": query,
                "vanilla": {"answer": ans_v, "latency": lat_v, "score": sv},
                "advanced": {"answer": ans_a, "latency": lat_a, "score": sa, "router": {
                    "doc_ids": decision.doc_ids,
                    "doc_type": decision.doc_type,
                    "section_ids": decision.section_ids,
                }},
                "winner": winner,
                "judge": judge,
            }
        )

    # Summary
    n = len(questions)
    print("\n" + "=" * 40)
    print("       BENCHMARK SUMMARY")
    print("=" * 40)
    print(f"Advanced Wins: {stats['Advanced']}")
    print(f"Vanilla Wins:  {stats['Vanilla']}")
    print(f"Ties:          {stats['Tie']}")
    print(f"\nAdvanced Win Rate: {(stats['Advanced']/n)*100:.1f}%")
    print("=" * 40)

    out_dir = Path(s.eval_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
