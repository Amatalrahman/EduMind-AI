"""
Reciprocal Rank Fusion (RRF).
Merges multiple ranked lists into a single fused ranking.

Formula: score(d) = Σ  1 / (k + rank_i(d))
         where rank_i(d) is the position of document d in ranked list i (1-indexed),
         and k is a constant (default 60, as per the original paper).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RRFResult:
    chroma_id: str
    rrf_score: float
    source_ranks: dict[str, int] = field(default_factory=dict)  # {'dense': 2, 'bm25': 5}


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    list_names: list[str] | None = None,
    k: int = 60,
) -> list[RRFResult]:
    """
    Fuse multiple ranked lists via RRF.

    Args:
        ranked_lists:  Each inner list is a ranked list of chroma_ids (best first).
        list_names:    Optional name for each list (for debugging).
        k:             RRF constant (default 60).

    Returns:
        List of RRFResult objects sorted by descending rrf_score.
    """
    if list_names is None:
        list_names = [str(i) for i in range(len(ranked_lists))]

    scores: dict[str, float] = {}
    source_ranks: dict[str, dict[str, int]] = {}

    for name, ranked in zip(list_names, ranked_lists):
        for rank, doc_id in enumerate(ranked, start=1):
            contribution = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + contribution
            if doc_id not in source_ranks:
                source_ranks[doc_id] = {}
            source_ranks[doc_id][name] = rank

    fused = [
        RRFResult(
            chroma_id=doc_id,
            rrf_score=score,
            source_ranks=source_ranks.get(doc_id, {}),
        )
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return fused
