"""
Chroma vector store wrapper.
One persistent collection per subject: "subject_{subject_id}".
"""

from __future__ import annotations

import logging
import streamlit as st
import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_chroma_client() -> chromadb.PersistentClient:
    """Return a singleton Chroma persistent client."""
    logger.info("Initialising Chroma persistent client at: %s", CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client


def _collection_name(subject_id: int) -> str:
    return f"subject_{subject_id}"


def get_or_create_collection(subject_id: int) -> chromadb.Collection:
    """Return (or create) the Chroma collection for the given subject."""
    client = get_chroma_client()
    name = _collection_name(subject_id)
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def upsert_chunks(
    subject_id: int,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """Upsert a batch of chunks into the subject's Chroma collection."""
    collection = get_or_create_collection(subject_id)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.debug("Upserted %d chunks into collection %s", len(ids), _collection_name(subject_id))


def query_collection(
    subject_id: int,
    query_embedding: list[float],
    n_results: int = 10,
) -> dict:
    """
    Query the subject's collection with a pre-computed embedding.

    Returns:
        Chroma query result dict with keys: ids, distances, documents, metadatas.
    """
    collection = get_or_create_collection(subject_id)
    count = collection.count()
    if count == 0:
        return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
    n_results = min(n_results, count)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def collection_count(subject_id: int) -> int:
    """Return the number of vectors stored for a subject."""
    try:
        return get_or_create_collection(subject_id).count()
    except Exception:
        return 0
