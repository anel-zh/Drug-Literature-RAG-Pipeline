from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Returns float32 embeddings shape (N, D)
        """
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype("float32")