# src/media/mmr.py
from __future__ import annotations
from typing import List, Sequence
import numpy as np

def maximal_marginal_relevance(
    query_vec: np.ndarray,
    doc_vecs: Sequence[np.ndarray],
    k: int,
    lambda_mult: float = 0.65,
) -> List[int]:
    if not doc_vecs:
        return []
    q = _norm(query_vec)
    X = np.vstack([_norm(d) for d in doc_vecs])
    sim_to_query = X @ q
    selected: List[int] = []
    candidates = set(range(len(doc_vecs)))
    while candidates and len(selected) < k:
        if not selected:
            i = int(np.argmax(sim_to_query[list(candidates)]))
            idx = list(candidates)[i]
            selected.append(idx)
            candidates.remove(idx)
            continue
        selected_vecs = X[selected]
        diversity = X[list(candidates)] @ selected_vecs.T
        max_div = diversity.max(axis=1)
        scores = lambda_mult * sim_to_query[list(candidates)] - (1 - lambda_mult) * max_div
        idx_local = int(np.argmax(scores))
        idx = list(candidates)[idx_local]
        selected.append(idx)
        candidates.remove(idx)
    return selected

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) or 1.0
    return v / n
