"""
Embedding wrapper for BAAI/bge-m3 via sentence-transformers.
The model is loaded once and cached as a Streamlit resource.
"""

from __future__ import annotations

import logging
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder() -> SentenceTransformer:
    """Load and cache the bge-m3 model. Called once per Streamlit session cluster."""
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings.
    Returns a list of normalized float vectors (L2 norm = 1).
    """
    model = get_embedder()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
