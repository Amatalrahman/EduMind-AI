"""
Flashcards tab — generate and review flashcards via Groq.
"""

from __future__ import annotations

import json
import logging

import streamlit as st

from db.database import (
    get_all_subjects,
    get_documents_by_subject,
    insert_flashcard,
    get_flashcards_by_subject,
)
from retrieval.hybrid_retriever import retrieve
from llm import groq_client

logger = logging.getLogger(__name__)

FLASHCARD_SYSTEM = """You are a flashcard generator for students.
Create concise, effective flashcards (question on front, answer on back).
Return ONLY a valid JSON object with this exact schema:
{
  "flashcards": [
    {
      "question": "What is...?",
      "answer": "...",
      "source_page": 3
    }
  ]
}
"""

FLASHCARD_PROMPT = """Generate {n} flashcards from the following lecture content.
Focus on key terms, definitions, concepts, and important facts.

{context}

Return exactly {n} flashcards in the required JSON schema.
"""


def _generate_flashcards(subject_id: int, query: str, n: int) -> list[dict]:
    chunks = retrieve(query=query, subject_id=subject_id, top_k=10)
    if not chunks:
        return []
    context = "\n\n".join(
        f"[{c.filename}, p.{c.page_number}]\n{c.text}" for c in chunks
    )
    try:
        from llm import QuotaExhaustedError
        raw = groq_client.generate_text(
            prompt=FLASHCARD_PROMPT.format(n=n, context=context),
            system_prompt=FLASHCARD_SYSTEM,
            max_tokens=2048,
            temperature=0.3,
            json_mode=True,
        )
        data = json.loads(raw)
        return data.get("flashcards", [])
    except QuotaExhaustedError as exc:
        st.error(f"API Quota Exhausted: {exc}\n\nPlease try again later. Other non-AI features remain available.")
        return []
    except Exception as exc:
        logger.error("Flashcard JSON parse error: %s", exc)
        st.error(f"Failed to parse flashcard JSON: {exc}")
        return []


def render_flashcards_tab():
    st.header("🃏 Flashcards")
    st.caption("Generate and review AI-created flashcards powered by Groq.")

    subjects = get_all_subjects()
    if not subjects:
        st.info("No subjects yet. Upload documents first.")
        return

    subject_names = [s["name"] for s in subjects]
    selected_name = st.selectbox("Subject", options=subject_names, key="fc_subject")
    selected_subject = next(s for s in subjects if s["name"] == selected_name)
    subject_id = selected_subject["id"]

    tab_gen, tab_review = st.tabs(["✨ Generate", "📖 Review Saved"])

    # ── Generate tab ───────────────────────────────────────────────────────────
    with tab_gen:
        col1, col2 = st.columns(2)
        with col1:
            topic_query = st.text_input(
                "Topic / focus area",
                placeholder="e.g. mitosis, gradient descent…",
                key="fc_topic",
            )
        with col2:
            n_cards = st.slider("Number of flashcards", 5, 30, 10, key="fc_n")

        query = topic_query.strip() if topic_query.strip() else "key concepts and definitions"

        if st.button("🎴 Generate Flashcards", key="gen_fc_btn", type="primary"):
            with st.spinner("Generating flashcards…"):
                cards = _generate_flashcards(subject_id, query, n_cards)

            if not cards:
                st.warning("Could not generate flashcards. Make sure documents are uploaded.")
                return

            # Save to DB
            for card in cards:
                insert_flashcard(
                    subject_id=subject_id,
                    question=card["question"],
                    answer=card["answer"],
                    source_page=card.get("source_page"),
                )

            st.session_state["current_flashcards"] = cards
            st.session_state["fc_index"] = 0
            st.session_state["fc_flipped"] = False
            st.success(f"Generated and saved {len(cards)} flashcards!")

        # Flashcard viewer
        cards = st.session_state.get("current_flashcards", [])
        if cards:
            st.divider()
            idx = st.session_state.get("fc_index", 0)
            flipped = st.session_state.get("fc_flipped", False)
            card = cards[idx]

            st.markdown(f"**Card {idx+1} of {len(cards)}**")
            card_container = st.container(border=True)
            with card_container:
                if not flipped:
                    st.markdown(f"### ❓ {card['question']}")
                    st.caption(f"Source: p.{card.get('source_page', '?')}")
                else:
                    st.markdown(f"### ✅ {card['answer']}")

            col_a, col_b, col_c = st.columns(3)
            if col_a.button("⬅️ Previous", key="fc_prev"):
                st.session_state["fc_index"] = max(0, idx - 1)
                st.session_state["fc_flipped"] = False
                st.rerun()
            if col_b.button("🔄 Flip", key="fc_flip"):
                st.session_state["fc_flipped"] = not flipped
                st.rerun()
            if col_c.button("Next ➡️", key="fc_next"):
                st.session_state["fc_index"] = min(len(cards) - 1, idx + 1)
                st.session_state["fc_flipped"] = False
                st.rerun()

    # ── Review saved tab ───────────────────────────────────────────────────────
    with tab_review:
        saved = get_flashcards_by_subject(subject_id)
        if not saved:
            st.info("No saved flashcards for this subject yet.")
            return
        st.write(f"**{len(saved)} saved flashcards**")
        for card in saved:
            with st.expander(f"❓ {card['question']}"):
                st.markdown(f"**Answer:** {card['answer']}")
                if card.get("source_page"):
                    st.caption(f"Source page: {card['source_page']}")
