"""
Summaries tab — per-lecture and per-subject summaries via RAG.
"""

from __future__ import annotations

import streamlit as st

from db.database import get_all_subjects, get_documents_by_subject
from rag.qa_chain import summarize


def render_summaries_tab():
    st.header("📝 Summaries")
    st.caption("Generate AI summaries from your uploaded lecture materials using RAG (not raw text).")

    subjects = get_all_subjects()
    if not subjects:
        st.info("No subjects yet. Upload documents first.")
        return

    subject_names = [s["name"] for s in subjects]
    selected_name = st.selectbox("Select subject", options=subject_names, key="sum_subject")
    selected_subject = next(s for s in subjects if s["name"] == selected_name)
    subject_id = selected_subject["id"]

    mode = st.radio(
        "Summary scope",
        options=["Whole subject (cross-lecture)", "Single lecture"],
        horizontal=True,
        key="sum_mode",
    )

    if mode == "Single lecture":
        docs = get_documents_by_subject(subject_id)
        if not docs:
            st.info("No documents found for this subject.")
            return
        doc_names = [d["filename"] for d in docs]
        selected_doc_name = st.selectbox("Select lecture", options=doc_names, key="sum_doc")
        title = selected_doc_name
        query = f"summarize the main topics and key concepts in {selected_doc_name}"
    else:
        title = selected_name
        query = "summarize all main topics and key concepts"

    if st.button("✨ Generate Summary", key="gen_summary_btn", type="primary"):
        with st.spinner("Retrieving relevant chunks and summarizing…"):
            summary_text = summarize(
                subject_id=subject_id,
                title=title,
                query=query,
                top_k=12,
            )
        st.markdown("---")
        st.subheader(f"Summary: {title}")
        st.markdown(summary_text)
        st.download_button(
            "⬇️ Download summary",
            data=summary_text,
            file_name=f"summary_{title.replace(' ', '_')}.md",
            mime="text/markdown",
        )
