"""
Hybrid retriever: combines dense (Chroma) + sparse (BM25) via RRF.
Returns the top-k most relevant chunks with full metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import TOP_K_DENSE, TOP_K_BM25, TOP_K_FINAL, RRF_K
from db.database import get_chunk_by_chroma_id, get_document_by_id
from indexing.embedder import embed_query
from indexing.vector_store import query_collection
from indexing.bm25_index import search_bm25
from retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned by the hybrid retriever, enriched with source metadata."""
    chroma_id: str
    text: str
    page_number: int
    chunk_index: int
    doc_id: int
    filename: str
    subject_id: int
    rrf_score: float


def retrieve(
    query: str,
    subject_id: int,
    top_k: int = TOP_K_FINAL,
) -> list[RetrievedChunk]:
    """
    Run hybrid retrieval for a query within a subject.

    Pipeline:
        1. Embed query with bge-m3
        2. Dense search in Chroma (cosine similarity)
        3. Sparse search via BM25
        4. Fuse with Reciprocal Rank Fusion
        5. Enrich top-k results with SQLite metadata (filename, page_number)

    Args:
        query:      The student's question.
        subject_id: Which subject's chunks to search.
        top_k:      Number of chunks to return after fusion.

    Returns:
        List of RetrievedChunk, best first.
    """
    # ── Step 1: Embed query ────────────────────────────────────────────────────
    query_embedding = embed_query(query)

    # ── Step 2: Dense retrieval ────────────────────────────────────────────────
    dense_result = query_collection(
        subject_id=subject_id,
        query_embedding=query_embedding,
        n_results=TOP_K_DENSE,
    )
    dense_ids: list[str] = dense_result["ids"][0] if dense_result["ids"] else []

    # ── Step 3: BM25 retrieval ─────────────────────────────────────────────────
    bm25_results = search_bm25(subject_id=subject_id, query=query, top_k=TOP_K_BM25)
    bm25_ids: list[str] = [r.chroma_id for r in bm25_results]

    logger.debug(
        "Dense hits: %d | BM25 hits: %d",
        len(dense_ids),
        len(bm25_ids),
    )

    # ── Step 4: RRF fusion ─────────────────────────────────────────────────────
    fused = reciprocal_rank_fusion(
        ranked_lists=[dense_ids, bm25_ids],
        list_names=["dense", "bm25"],
        k=RRF_K,
    )
    top_fused = fused[:top_k]

    # ── Step 5: Enrich with metadata ───────────────────────────────────────────
    results: list[RetrievedChunk] = []
    for item in top_fused:
        chunk_row = get_chunk_by_chroma_id(item.chroma_id)
        if chunk_row is None:
            logger.warning("Chroma id %s not found in SQLite; skipping", item.chroma_id)
            continue
        doc_row = get_document_by_id(chunk_row["doc_id"])
        if doc_row is None:
            continue

        results.append(
            RetrievedChunk(
                chroma_id=item.chroma_id,
                text=chunk_row["text"],
                page_number=chunk_row["page_number"],
                chunk_index=chunk_row["chunk_index"],
                doc_id=chunk_row["doc_id"],
                filename=doc_row["filename"],
                subject_id=chunk_row["subject_id"],
                rrf_score=item.rrf_score,
            )
        )

    logger.info("Hybrid retriever returned %d chunks for query: %.60s…", len(results), query)
    return results
