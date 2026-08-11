"""
Upload tab — handles subject management and file ingestion.
Pipeline: Upload file → parse → chunk → embed → store in Chroma + SQLite + rebuild BM25.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import streamlit as st

from config import UPLOADS_DIR
from db.database import (
    create_subject,
    get_all_subjects,
    insert_document,
    insert_chunks_batch,
    get_documents_by_subject,
)
from ingestion.pdf_parser import parse_pdf
from ingestion.pptx_parser import parse_pptx
from ingestion.chunker import chunk_pages
from indexing.embedder import embed_texts
from indexing.vector_store import upsert_chunks, collection_count
from indexing.bm25_index import build_bm25_index

logger = logging.getLogger(__name__)


def _save_upload(uploaded_file) -> Path:
    """Persist the uploaded bytes to disk and return the path."""
    dest = UPLOADS_DIR / uploaded_file.name
    # Avoid overwrite collisions
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        dest = UPLOADS_DIR / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def _ingest_file(file_path: Path, file_type: str, doc_id: int, subject_id: int) -> int:
    """
    Full ingestion pipeline for one file.
    Returns the number of chunks stored.
    """
    # 1. Parse
    if file_type == "pdf":
        pages = parse_pdf(file_path)
    else:
        pages = parse_pptx(file_path)

    # 2. Chunk
    chunks = chunk_pages(pages)
    if not chunks:
        st.warning("No text content found in the uploaded file.")
        return 0

    # 3. Embed
    texts = [c.text for c in chunks]
    with st.spinner(f"Embedding {len(chunks)} chunks (this may take a moment)…"):
        embeddings = embed_texts(texts)

    # 4. Build Chroma IDs and metadata
    chroma_ids = [f"doc{doc_id}_chunk{c.chunk_index}" for c in chunks]
    metadatas = [
        {
            "doc_id": doc_id,
            "subject_id": subject_id,
            "page_number": c.page_number,
            "filename": file_path.name,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    # 5. Upsert into Chroma
    upsert_chunks(
        subject_id=subject_id,
        ids=chroma_ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    # 6. Store chunks in SQLite
    sqlite_rows = [
        (doc_id, subject_id, c.chunk_index, c.text, c.page_number, chroma_id)
        for c, chroma_id in zip(chunks, chroma_ids)
    ]
    insert_chunks_batch(sqlite_rows)

    # 7. Rebuild BM25 index for this subject
    with st.spinner("Rebuilding BM25 index…"):
        build_bm25_index(subject_id)

    return len(chunks)


def render_upload_tab():
    st.header("📂 Upload Lecture Materials")

    # ── Subject management ─────────────────────────────────────────────────────
    st.subheader("1. Choose or Create a Subject")
    subjects = get_all_subjects()
    subject_names = [s["name"] for s in subjects]

    col1, col2 = st.columns([3, 1])
    with col1:
        new_subject = st.text_input(
            "New subject name",
            placeholder="e.g. Machine Learning, Organic Chemistry…",
            key="new_subject_input",
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("➕ Create", key="create_subject_btn"):
            if new_subject.strip():
                sid = create_subject(new_subject.strip())
                st.success(f"Subject '{new_subject.strip()}' created (id={sid}).")
                st.rerun()
            else:
                st.warning("Please enter a subject name.")

    if subject_names:
        selected_subject_name = st.selectbox(
            "Select subject to upload into",
            options=subject_names,
            key="selected_subject",
        )
        selected_subject = next(s for s in subjects if s["name"] == selected_subject_name)
        subject_id = selected_subject["id"]
    else:
        st.info("Create a subject above before uploading files.")
        return

    st.divider()

    # ── File upload ────────────────────────────────────────────────────────────
    st.subheader("2. Upload File")
    uploaded_file = st.file_uploader(
        "Upload a PDF or PowerPoint lecture",
        type=["pdf", "pptx"],
        key="file_uploader",
    )

    if uploaded_file is not None:
        file_type = uploaded_file.name.rsplit(".", 1)[-1].lower()
        st.write(f"**File:** {uploaded_file.name}  |  **Type:** {file_type.upper()}  |  **Size:** {uploaded_file.size / 1024:.1f} KB")

        if st.button("🚀 Ingest File", key="ingest_btn", type="primary"):
            with st.status("Ingesting file…", expanded=True) as status:
                st.write(f"📄 Saving file to disk…")
                file_path = _save_upload(uploaded_file)

                st.write("📝 Registering document in database…")
                doc_id = insert_document(
                    subject_id=subject_id,
                    filename=uploaded_file.name,
                    file_type=file_type,
                    file_path=str(file_path),
                )

                st.write("🔍 Parsing, chunking, and indexing…")
                try:
                    n_chunks = _ingest_file(
                        file_path=file_path,
                        file_type=file_type,
                        doc_id=doc_id,
                        subject_id=subject_id,
                    )
                    status.update(label=f"✅ Done! {n_chunks} chunks indexed.", state="complete")
                    st.success(f"Successfully indexed **{n_chunks}** chunks from **{uploaded_file.name}**.")
                    st.info(f"Total vectors in subject '{selected_subject_name}': {collection_count(subject_id)}")
                except Exception as exc:
                    status.update(label="❌ Ingestion failed.", state="error")
                    st.error(f"Error during ingestion: {exc}")
                    logger.exception("Ingestion error for doc_id=%d", doc_id)

    st.divider()

    # ── Existing documents ─────────────────────────────────────────────────────
    st.subheader(f"3. Documents in '{selected_subject_name}'")
    docs = get_documents_by_subject(subject_id)
    if docs:
        for doc in docs:
            col_a, col_b, col_c = st.columns([4, 2, 2])
            col_a.write(f"📄 **{doc['filename']}**")
            col_b.write(doc["file_type"].upper())
            col_c.write(doc["upload_date"][:10])
        st.caption(f"Total vectors indexed for this subject: **{collection_count(subject_id)}**")
    else:
        st.info("No documents uploaded yet for this subject.")
