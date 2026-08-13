"""
Chat / Q&A tab.
Hybrid RAG Q&A with citation display and short-term memory.
"""

from __future__ import annotations

import streamlit as st

from db.database import get_all_subjects
from rag.qa_chain import answer_question, ChatTurn

_HISTORY_KEY = "chat_history"
_SUBJECT_KEY = "chat_subject_id"


def _get_history() -> list[ChatTurn]:
    return st.session_state.get(_HISTORY_KEY, [])


def _add_to_history(role: str, content: str):
    if _HISTORY_KEY not in st.session_state:
        st.session_state[_HISTORY_KEY] = []
    st.session_state[_HISTORY_KEY].append(ChatTurn(role=role, content=content))


def render_chat_tab():
    st.header("💬 Chat with your Lectures")

    # ── Subject selector ───────────────────────────────────────────────────────
    subjects = get_all_subjects()
    if not subjects:
        st.warning("No subjects found. Please go to the **Upload** tab and create a subject first.")
        return

    subject_names = [s["name"] for s in subjects]
    col1, col2 = st.columns([4, 1])
    with col1:
        selected_name = st.selectbox(
            "Subject to query",
            options=subject_names,
            key="chat_subject_select",
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear chat", key="clear_chat_btn"):
            st.session_state[_HISTORY_KEY] = []
            st.rerun()

    selected_subject = next(s for s in subjects if s["name"] == selected_name)
    subject_id = selected_subject["id"]
    st.session_state[_SUBJECT_KEY] = subject_id

    st.caption(
        "Ask anything about your uploaded lectures. "
        "Every answer will cite its source with **[Lecture: …, p.X]**."
    )
    st.divider()

    # ── Chat history display ───────────────────────────────────────────────────
    history = _get_history()
    for turn in history:
        with st.chat_message(turn.role):
            st.markdown(turn.content)

    # ── Input ──────────────────────────────────────────────────────────────────
    if question := st.chat_input("Ask a question about your lecture materials…", key="chat_input"):
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)
        _add_to_history("user", question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching your lectures…"):
                try:
                    from llm import QuotaExhaustedError
                    result = answer_question(
                        question=question,
                        subject_id=subject_id,
                        chat_history=history,
                    )
                except QuotaExhaustedError as exc:
                    st.error(f"API Quota Exhausted: {exc}\n\nPlease try again later. Other non-AI features remain available.")
                    st.stop()
                    
            st.markdown(result.answer)

            # Sources panel
            if result.sources:
                with st.expander("📚 Sources used", expanded=False):
                    for src in result.sources:
                        st.markdown(
                            f"- **{src['filename']}** — page {src['page']}\n"
                            f"  > *{src['snippet']}*"
                        )

        _add_to_history("assistant", result.answer)
        st.rerun()

