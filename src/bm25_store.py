from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from .tokenizer import tokenize

@dataclass
class BM25Result:
    idx: int
    score: float

class BM25Store:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 10) -> List[BM25Result]:
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [BM25Result(idx=i, score=float(s)) for i, s in ranked if s > 0.0]
    
    def search_in_subset(self, query: str, subset_idxs: List[int], top_k: int = 10) -> List[int]:
        """Finds the best matches specifically within a set of allowed chunks."""
        if not subset_idxs:
            return []
            
        # 1. Use the SAME tokenizer as the main search
        q_tokens = tokenize(query)
        
        # 2. Get scores for the WHOLE corpus 
        all_scores = self.bm25.get_scores(q_tokens)
        
        # 3. Pick out only the scores for allowed indices
        subset_scored_idxs = []
        for i in subset_idxs:
            # i is the index in the original chunks list
            subset_scored_idxs.append((i, float(all_scores[i])))
            
        # 4. Sort only this subset by score descending
        ranked = sorted(subset_scored_idxs, key=lambda x: x[1], reverse=True)
        
        # 5. Return just the top_k indices
        return [idx for idx, score in ranked[:top_k]]
