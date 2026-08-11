"""
BM25 index manager.
Builds a per-subject BM25 index from all chunks stored in SQLite,
pickles it to disk so it survives Streamlit restarts.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from config import BM25_DIR, TOP_K_BM25
from db.database import get_chunks_by_subject

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    chroma_id: str
    score: float
    rank: int


def _index_path(subject_id: int) -> Path:
    return BM25_DIR / f"bm25_subject_{subject_id}.pkl"


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer (fast; good enough for BM25)."""
    return text.lower().split()


def build_bm25_index(subject_id: int) -> None:
    """
    Build and persist a BM25 index for all chunks belonging to a subject.
    Always rebuilds from the current SQLite state.
    """
    chunks = get_chunks_by_subject(subject_id)
    if not chunks:
        logger.warning("No chunks found for subject_id=%d — skipping BM25 build", subject_id)
        return

    corpus = [chunk["text"] for chunk in chunks]
    chroma_ids = [chunk["chroma_id"] for chunk in chunks]
    tokenized = [_tokenize(text) for text in corpus]

    bm25 = BM25Okapi(tokenized)

    payload = {"bm25": bm25, "chroma_ids": chroma_ids}
    with open(_index_path(subject_id), "wb") as f:
        pickle.dump(payload, f)

    logger.info("Built BM25 index for subject_id=%d  (%d docs)", subject_id, len(corpus))


def _load_index(subject_id: int) -> dict | None:
    path = _index_path(subject_id)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def search_bm25(subject_id: int, query: str, top_k: int = TOP_K_BM25) -> list[BM25Result]:
    """
    Search BM25 index for a subject.
    If no index exists, builds it first.

    Returns:
        Ranked list of BM25Result (highest score first).
    """
    payload = _load_index(subject_id)
    if payload is None:
        logger.info("BM25 index not found for subject_id=%d — building now…", subject_id)
        build_bm25_index(subject_id)
        payload = _load_index(subject_id)
        if payload is None:
            return []

    bm25: BM25Okapi = payload["bm25"]
    chroma_ids: list[str] = payload["chroma_ids"]

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Sort by descending score and take top-k
    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    results = [
        BM25Result(chroma_id=chroma_ids[idx], score=float(score), rank=rank + 1)
        for rank, (idx, score) in enumerate(ranked)
        if score > 0  # ignore zero-score docs
    ]
    return results
