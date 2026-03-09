from __future__ import annotations

import json
from pathlib import Path

from src.config import Settings
from src.pdf_loader import load_pdf_pages
from src.chunker import chunk_pages
from src.embedder import Embedder
from src.index_store import build_faiss_index, save_faiss_index, save_chunks_jsonl, save_meta


def make_doc_id_from_filename(p: Path) -> str:
    # Example: HADLIMA_FDA_LABEL.pdf -> HADLIMA_FDA_LABEL
    return p.stem


def infer_doc_type(doc_id: str) -> str:
    if "FDA" in doc_id.upper() or "LABEL" in doc_id.upper():
        return "fda_label"
    return "paper"


def main():
    s = Settings()

    pdf_dir = Path(s.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {pdf_dir}")

    all_pages = []
    docs = []

    for pdf_path in pdfs:
        doc_id = make_doc_id_from_filename(pdf_path)
        doc_type = infer_doc_type(doc_id)

        pages = load_pdf_pages(pdf_path, doc_id=doc_id, doc_type=doc_type)
        # convert to dicts expected by chunker
        page_dicts = [
            {
                "doc_id": p["doc_id"], 
                "doc_type": p["doc_type"], 
                "page": p["page"], 
                "text": p["text"]
            } 
            for p in pages
        ]
        all_pages.extend(page_dicts)

        # aliases used by router (scales to many docs)
        aliases = [doc_id.lower()]
        # if doc_id begins with drug name e.g., HADLIMA_...
        aliases.append(doc_id.split("_")[0].lower())
        docs.append({"doc_id": doc_id, "doc_type": doc_type, "aliases": sorted(set(aliases))})

    chunks = chunk_pages(all_pages, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
    texts = [c.text for c in chunks]

    embedder = Embedder(s.embedding_model)
    emb = embedder.embed_texts(texts, batch_size=s.embed_batch_size)

    index = build_faiss_index(emb)

    # Save artifacts
    Path(s.index_dir).mkdir(parents=True, exist_ok=True)
    save_faiss_index(index, Path(s.faiss_index_path))
    save_chunks_jsonl(chunks, Path(s.chunks_path))

    meta = {
        "embedding_model": s.embedding_model,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "docs": docs,
        "num_docs": len(docs),
        "num_chunks": len(chunks),
    }
    save_meta(meta, Path(s.meta_path))

    print(f"[OK] Indexed {len(docs)} PDFs, {len(chunks)} chunks.")
    print(f"[OK] Saved: {s.faiss_index_path}")
    print(f"[OK] Saved: {s.chunks_path}")
    print(f"[OK] Saved: {s.meta_path}")


if __name__ == "__main__":
    main()