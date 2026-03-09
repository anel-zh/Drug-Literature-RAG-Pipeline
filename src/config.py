from dataclasses import dataclass

@dataclass
class Settings:
    pdf_dir: str = "data/pdfs"
    index_dir: str = "index"
    faiss_index_path: str = "index/faiss.index"
    chunks_path: str = "index/chunks.jsonl"
    meta_path: str = "index/meta.json"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embed_batch_size: int = 32

    chunk_size: int = 1200
    chunk_overlap: int = 250

    local_llm_model: str = "llama3.1:8b"
    llm_num_ctx: int = 4096
    answer_max_tokens: int = 350

    eval_questions_path: str = "data/eval/questions.jsonl"
    eval_out_dir: str = "data/eval/out"