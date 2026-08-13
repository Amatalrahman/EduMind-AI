"""
Quizzes tab — generate MCQ quizzes from lecture content via Groq.
"""

from __future__ import annotations

import json
import logging

import streamlit as st

from db.database import (
    get_all_subjects,
    get_documents_by_subject,
    insert_quiz_result,
    upsert_study_log,
)
from retrieval.hybrid_retriever import retrieve
from llm import groq_client

logger = logging.getLogger(__name__)

QUIZ_SYSTEM = """You are a university-level quiz generator.
Generate multiple-choice questions from the provided lecture content.
Return ONLY a valid JSON object with this exact schema:
{
  "questions": [
    {
      "question": "...",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": "A",
      "explanation": "...",
      "source_page": 5
    }
  ]
}
Do NOT include any text outside the JSON object.
"""

QUIZ_PROMPT = """Generate {n} multiple-choice questions from the following lecture content.
Questions must test deep understanding, not superficial recall.

{context}

Return exactly {n} questions in the required JSON schema.
"""


def _generate_quiz(subject_id: int, query: str, n_questions: int) -> list[dict]:
    chunks = retrieve(query=query, subject_id=subject_id, top_k=10)
    if not chunks:
        return []

    context = "\n\n".join(
        f"[{c.filename}, p.{c.page_number}]\n{c.text}" for c in chunks
    )

    try:
        from llm import QuotaExhaustedError
        raw = groq_client.generate_text(
            prompt=QUIZ_PROMPT.format(n=n_questions, context=context),
            system_prompt=QUIZ_SYSTEM,
            max_tokens=2048,
            temperature=0.4,
            json_mode=True,
        )
        data = json.loads(raw)
        return data.get("questions", [])
    except QuotaExhaustedError as exc:
        st.error(f"API Quota Exhausted: {exc}\n\nPlease try again later. Other non-AI features remain available.")
        return []
    except Exception as exc:
        logger.error("Quiz JSON parse error: %s", exc)
        st.error(f"Failed to parse quiz JSON: {exc}")
        return []


def render_quizzes_tab():
    st.header("🧠 Quizzes")
    st.caption("AI-generated MCQ quizzes from your lecture materials. Powered by Groq for fast generation.")

    subjects = get_all_subjects()
    if not subjects:
        st.info("No subjects yet. Upload documents first.")
        return

    subject_names = [s["name"] for s in subjects]
    selected_name = st.selectbox("Subject", options=subject_names, key="quiz_subject")
    selected_subject = next(s for s in subjects if s["name"] == selected_name)
    subject_id = selected_subject["id"]

    col1, col2 = st.columns(2)
    with col1:
        topic_query = st.text_input(
            "Topic / focus area (optional)",
            placeholder="e.g. neural networks, photosynthesis…",
            key="quiz_topic",
        )
    with col2:
        n_questions = st.slider("Number of questions", 3, 15, 5, key="quiz_n")

    query = topic_query.strip() if topic_query.strip() else "key concepts and definitions"

    if st.button("🎯 Generate Quiz", key="gen_quiz_btn", type="primary"):
        with st.spinner("Generating quiz with Groq…"):
            questions = _generate_quiz(subject_id, query, n_questions)

        if not questions:
            st.warning("Could not generate questions. Make sure documents are uploaded and indexed.")
            return

        st.session_state["quiz_questions"] = questions
        st.session_state["quiz_answers"] = {}
        st.session_state["quiz_submitted"] = False
        st.session_state["quiz_subject_id"] = subject_id
        st.session_state["saved_quiz_topic"] = query

    # ── Display quiz ───────────────────────────────────────────────────────────
    questions = st.session_state.get("quiz_questions", [])
    if not questions:
        return

    st.divider()
    st.subheader(f"Quiz: {selected_name} — {query}")

    with st.form("quiz_form"):
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            options = q.get("options", [])
            answer = st.radio(
                f"Q{i+1}",
                options=options,
                key=f"quiz_q_{i}",
                label_visibility="collapsed",
            )
            st.caption(f"Source: p.{q.get('source_page', '?')}")
            st.markdown("---")

        submitted = st.form_submit_button("✅ Submit Quiz")

    if submitted:
        score = 0
        for i, q in enumerate(questions):
            user_choice = st.session_state.get(f"quiz_q_{i}", "")
            correct = q.get("correct_answer", "")
            # Match by letter prefix
            is_correct = user_choice.startswith(correct) if user_choice else False
            if is_correct:
                score += 1
            with st.expander(f"Q{i+1}: {'✅' if is_correct else '❌'} {q['question']}"):
                st.markdown(f"**Your answer:** {user_choice}")
                st.markdown(f"**Correct answer:** {correct}")
                st.markdown(f"**Explanation:** {q.get('explanation', 'N/A')}")

        accuracy = score / len(questions)
        st.success(f"**Score: {score}/{len(questions)} ({accuracy*100:.0f}%)**")

        # Save to DB
        insert_quiz_result(
            subject_id=st.session_state.get("quiz_subject_id", subject_id),
            score=score,
            total=len(questions),
        )
        topic = st.session_state.get("saved_quiz_topic", query)
        upsert_study_log(
            subject_id=st.session_state.get("quiz_subject_id", subject_id),
            topic=topic,
            quiz_accuracy=accuracy,
        )
        st.info("Results saved to your study log.")
