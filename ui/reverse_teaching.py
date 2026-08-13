"""
Reverse-Teaching Challenge tab.
The AI pretends to know nothing and asks the student to explain a topic.
It scores each explanation (0-100), surfaces knowledge gaps, and asks a follow-up.
After 3 rounds the final average score is shown as a summary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

from db.database import get_all_subjects
from retrieval.hybrid_retriever import retrieve
from llm.gemini_client import generate_text

logger = logging.getLogger(__name__)

# ── Paths to avatar images ─────────────────────────────────────────────────────
_FEELINGS_DIR = Path(__file__).parent.parent / "feelings"


def _avatar(score: float | None) -> str:
    """Return the path string for the reaction image based on score."""
    if score is None:
        return str(_FEELINGS_DIR / "search.png")
    if score < 40:
        return str(_FEELINGS_DIR / "confused.png")
    if score < 75:
        return str(_FEELINGS_DIR / "thinking.png")
    if score < 90:
        return str(_FEELINGS_DIR / "excited.png")
    # impressed is .jpg in this project
    impressed = _FEELINGS_DIR / "impressed.png"
    if not impressed.exists():
        impressed = _FEELINGS_DIR / "impressed.jpg"
    return str(impressed)


# ── Prompts ────────────────────────────────────────────────────────────────────

_QUESTION_SYSTEM = (
    "You are a curious student who knows absolutely nothing about this topic. "
    "Ask exactly ONE short, genuine, beginner-level question. "
    "Every round you must ask about a DIFFERENT aspect — do not repeat or paraphrase "
    "a question that was already asked. "
    "Return ONLY the question text, with no preamble, no numbering, no quotes."
)

_QUESTION_PROMPT = (
    "Topic context from lecture notes:\n__CTX__\n\n"
    "__AVOID__"
    "Ask me ONE beginner question about a fresh angle of this topic so I can teach it to you."
)

_EVAL_SYSTEM = (
    "You are an expert academic evaluator who grades student explanations fairly and generously. "
    "The student may use informal language, analogies, or a different structure than the textbook — "
    "that is fine. Credit any correct idea regardless of phrasing.\n\n"
    "Score the explanation on FOUR dimensions (each 0-25), then sum them for a total out of 100:\n"
    "  - Accuracy (0-25):     Are the facts and concepts stated correctly?"
    " Penalise only clear factual errors, not imprecise wording.\n"
    "  - Completeness (0-25): Does the answer cover the key points needed to answer the question?"
    " Partial coverage gets partial credit.\n"
    "  - Clarity (0-25):      Is the explanation understandable and logically structured?"
    " Simple but clear language scores well.\n"
    "  - Depth (0-25):        Does the student show understanding beyond surface recall?"
    " Examples, mechanisms, or 'why' earn higher marks.\n\n"
    "IMPORTANT: Almost no response deserves a perfect 100 or a flat 0. "
    "A partially correct answer should score 35-65. A mostly correct answer 65-85. "
    "A very good answer 85-95. Reserve 96-100 for genuinely complete and insightful answers.\n\n"
    "Return ONLY a valid JSON object — no markdown, no extra text — with EXACTLY this schema:\n"
    '{"score": <integer 0-100>, '
    '"breakdown": {"accuracy": <0-25>, "completeness": <0-25>, "clarity": <0-25>, "depth": <0-25>}, '
    '"gaps": [<string>, <string>], '
    '"ideal_answer": "<string>", '
    '"follow_up_question": "<string>"}\n\n'
    "gaps: 1-2 specific things missing or wrong. Empty list [] if nothing significant is missing.\n"
    "ideal_answer: A concise, clear model answer for THIS specific question (3-6 sentences). "
    "Write it as if explaining to the same beginner student.\n"
    "follow_up_question: A question that probes a gap or goes one level deeper."
)

_EVAL_PROMPT = (
    "The question asked to the student:\n__Q__\n\n"
    "Relevant topic context from lecture notes:\n__CTX__\n\n"
    'The student\'s explanation:\n"""__EXP__"""\n\n'
    "Evaluate the explanation against the question and context, then return the JSON."
)

# ── Session-state helpers ──────────────────────────────────────────────────────
_SK = "rt_"  # prefix for all reverse-teaching session keys


def _ss(key: str, default=None):
    return st.session_state.get(_SK + key, default)


def _set(key: str, val):
    st.session_state[_SK + key] = val


def _reset():
    """Clear all reverse-teaching session state."""
    for k in list(st.session_state.keys()):
        if k.startswith(_SK):
            del st.session_state[k]


# ── Core helpers ───────────────────────────────────────────────────────────────

def _build_context(subject_id: int, topic: str) -> str:
    """Retrieve relevant chunks and concatenate into a context string."""
    chunks = retrieve(query=topic, subject_id=subject_id, top_k=6)
    if not chunks:
        return ""
    return "\n\n".join(
        f"[{c.filename}, p.{c.page_number}]\n{c.text}" for c in chunks
    )


def _ask_opening_question(context: str, asked_questions: list[str]) -> str:
    """Have Gemini generate a beginner question, avoiding previously asked ones."""
    if asked_questions:
        avoid_clause = (
            "Questions already asked (do NOT repeat or paraphrase these):\n"
            + "\n".join(f"- {q}" for q in asked_questions)
            + "\n\n"
        )
    else:
        avoid_clause = ""
    # Use replace() instead of .format() so lecture content with { } never crashes
    prompt = (
        _QUESTION_PROMPT
        .replace("__CTX__", context)
        .replace("__AVOID__", avoid_clause)
    )
    return generate_text(
        prompt=prompt,
        system_instruction=_QUESTION_SYSTEM,
        temperature=0.85,
    ).strip()


def _evaluate_explanation(context: str, question: str, explanation: str) -> dict:
    """Multi-dimensional scoring: returns score, breakdown, gaps, ideal_answer, follow_up."""
    # Use replace() instead of .format() so lecture content with { } never crashes
    prompt = (
        _EVAL_PROMPT
        .replace("__Q__", question)
        .replace("__CTX__", context)
        .replace("__EXP__", explanation)
    )
    raw = generate_text(
        prompt=prompt,
        system_instruction=_EVAL_SYSTEM,
        temperature=0.25,
        max_tokens=2048,
    )
    # Strip possible markdown fences
    raw = raw.strip()
    for fence in ("```json", "```"):
        raw = raw.removeprefix(fence).removesuffix("```").strip()
    # Also extract JSON if wrapped in extra prose
    if raw and raw[0] != "{":
        start = raw.find("{")
        end   = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]
    try:
        data = json.loads(raw)
        score = max(0, min(100, int(data.get("score", 0))))
        return {
            "score": score,
            "breakdown": data.get("breakdown", {}),
            "gaps": data.get("gaps", []),
            "ideal_answer": data.get("ideal_answer", ""),
            "follow_up_question": data.get("follow_up_question", "Can you tell me more?"),
        }
    except Exception as exc:
        logger.error("Reverse-teaching eval JSON parse error: %s\nRaw output: %s", exc, raw)
        return {
            "score": 0,
            "breakdown": {},
            "gaps": [f"Evaluation parsing failed. Raw: {raw[:200]}"],
            "ideal_answer": "",
            "follow_up_question": "Let's try that again with a different topic.",
        }


# ── Main render function ───────────────────────────────────────────────────────

def render_reverse_teaching_tab():
    st.header("🎓 Explain It to Me")
    st.caption(
        "Teach the AI! Pick a topic, explain it round by round — "
        "the AI will score your understanding and ask follow-up questions."
    )

    # ── Subject + topic selector ───────────────────────────────────────────────
    subjects = get_all_subjects()
    if not subjects:
        st.info("No subjects yet. Go to **Upload** and add some lecture materials first.")
        return

    subject_names = [s["name"] for s in subjects]

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_name = st.selectbox(
            "Subject",
            options=subject_names,
            key="rt_subject_select",
            disabled=bool(_ss("started")),
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Reset", key="rt_reset_btn"):
            _reset()
            st.rerun()

    selected_subject = next(s for s in subjects if s["name"] == selected_name)
    subject_id = selected_subject["id"]

    topic = st.text_input(
        "Topic to teach",
        placeholder="e.g. photosynthesis, neural networks, TCP/IP…",
        key="rt_topic_input",
        disabled=bool(_ss("started")),
    )

    MAX_ROUNDS = 3

    # ── Start button ───────────────────────────────────────────────────────────
    if not _ss("started"):
        if st.button("🚀 Start Challenge", key="rt_start_btn", type="primary"):
            if not topic.strip():
                st.warning("Please enter a topic first.")
                st.stop()

            _set("started", True)
            _set("subject_id", subject_id)
            _set("topic", topic.strip())
            _set("round", 1)
            _set("scores", [])
            _set("messages", [])          # list of {role, content, score, breakdown, ideal_answer}
            _set("context", None)
            _set("awaiting_answer", False)
            _set("finished", False)
            _set("asked_questions", [])   # track questions to avoid repetition
            st.rerun()

    if not _ss("started"):
        return

    subject_id          = _ss("subject_id")
    topic               = _ss("topic")
    current_round: int  = _ss("round", 1)
    scores: list        = _ss("scores", [])
    messages: list      = _ss("messages", [])
    asked_questions: list = _ss("asked_questions", [])

    st.divider()
    st.markdown(
        f"**Topic:** `{topic}` &nbsp;|&nbsp; "
        f"**Round {min(current_round, MAX_ROUNDS)} / {MAX_ROUNDS}**"
    )

    # ── Retrieve context once per session ─────────────────────────────────────
    if _ss("context") is None:
        with st.chat_message("assistant", avatar="🤖"):
            st.image(_avatar(None), width=120)
            with st.spinner("🔍 Searching your lecture notes…"):
                context = _build_context(subject_id, topic)
            _set("context", context)

            if not context:
                st.warning(
                    "No content found for this topic. Make sure documents about "
                    f"**{topic}** are uploaded and indexed."
                )
                _reset()
                st.stop()

            # Generate opening question
            try:
                from llm import QuotaExhaustedError
                question = _ask_opening_question(context, [])
            except QuotaExhaustedError as exc:
                st.error(f"API Quota Exhausted: {exc}\n\nPlease try again later. Other non-AI features remain available.")
                _reset()
                st.stop()
            _set("current_question", question)
            _set("awaiting_answer", True)
            asked_questions.append(question)
            _set("asked_questions", asked_questions)
            messages.append({"role": "assistant", "content": question, "score": None,
                             "breakdown": None, "ideal_answer": None})
            _set("messages", messages)
            st.rerun()

    context = _ss("context", "")

    # ── Render all past messages ───────────────────────────────────────────────
    for msg in messages:
        role      = msg["role"]
        score     = msg.get("score")
        breakdown = msg.get("breakdown")
        ideal     = msg.get("ideal_answer")
        if role == "assistant":
            # Using a generic avatar for the bubble, placing the emotion image prominently inside
            with st.chat_message("assistant", avatar="🤖"):
                # Display large emotion icon
                img_path = _avatar(score)
                st.image(img_path, width=120)
                
                st.markdown(msg["content"])
                
                if score is not None:
                    badge = (
                        "🔴" if score < 40 else
                        "🟡" if score < 75 else
                        "🟢" if score < 90 else
                        "🌟"
                    )
                    st.markdown(f"{badge} **Score: {score}/100**")
                    st.progress(score / 100)
                    # Score breakdown
                    if breakdown:
                        bcols = st.columns(4)
                        labels = [("✅ Accuracy", "accuracy"), ("📋 Complete", "completeness"),
                                  ("💬 Clarity", "clarity"), ("🔬 Depth", "depth")]
                        for bcol, (label, key) in zip(bcols, labels):
                            with bcol:
                                st.metric(label, f"{breakdown.get(key, 0)}/25")
                    # Ideal answer reveal
                    if ideal:
                        with st.expander("📖 See ideal answer"):
                            st.markdown(ideal)
        else:
            with st.chat_message("user"):
                st.markdown(msg["content"])

    # ── Show final summary if finished ────────────────────────────────────────
    if _ss("finished"):
        avg = sum(scores) / len(scores) if scores else 0
        st.divider()
        st.subheader("🏆 Challenge Complete!")

        cols = st.columns(len(scores))
        for i, (col, s) in enumerate(zip(cols, scores)):
            with col:
                st.metric(f"Round {i + 1}", f"{s}/100")

        st.markdown(f"### Final Average Score: **{avg:.0f} / 100**")
        st.progress(avg / 100)
        return

    # ── Input box for student's explanation ───────────────────────────────────
    if _ss("awaiting_answer") and current_round <= MAX_ROUNDS:
        explanation = st.text_area(
            f"Your explanation (Round {current_round}/{MAX_ROUNDS})",
            placeholder="Explain the concept in your own words…",
            height=160,
            key=f"rt_answer_{current_round}",
        )

        if st.button(
            "📤 Submit Explanation",
            key=f"rt_submit_{current_round}",
            type="primary",
        ):
            if not explanation.strip():
                st.warning("Please write something first!")
                st.stop()

            # Record student message
            current_question = _ss("current_question", "")
            messages.append({"role": "user", "content": explanation.strip(),
                             "score": None, "breakdown": None, "ideal_answer": None})
            _set("messages", messages)
            _set("awaiting_answer", False)

            # Evaluate — pass the specific question so AI judges against it
            with st.chat_message("assistant", avatar="🤖"):
                st.image(_avatar(None), width=120)
                with st.spinner("🤔 Evaluating your explanation…"):
                    try:
                        from llm import QuotaExhaustedError
                        result = _evaluate_explanation(
                            context, current_question, explanation.strip()
                        )
                    except QuotaExhaustedError as exc:
                        st.error(f"API Quota Exhausted: {exc}\n\nPlease try again later. Other non-AI features remain available.")
                        st.stop()

            score        = result["score"]
            breakdown    = result["breakdown"]
            gaps         = result["gaps"]
            ideal_answer = result["ideal_answer"]
            follow_up    = result["follow_up_question"]

            # Build AI feedback text
            response_parts = []
            if gaps:
                gap_list = "\n".join(f"- {g}" for g in gaps)
                response_parts.append(f"**What I noticed was missing or unclear:**\n{gap_list}")
            if current_round < MAX_ROUNDS:
                response_parts.append(f"**Next question for you:** {follow_up}")
            else:
                response_parts.append("Great effort — that was the final round! 🎉")

            response_text = "\n\n".join(response_parts) if response_parts else "Good job!"
            messages.append({
                "role": "assistant",
                "content": response_text,
                "score": score,
                "breakdown": breakdown,
                "ideal_answer": ideal_answer,
            })
            scores.append(score)
            _set("messages", messages)
            _set("scores", scores)

            next_round = current_round + 1
            if next_round > MAX_ROUNDS:
                _set("finished", True)
                _set("round", next_round)
            else:
                _set("round", next_round)
                # Generate a genuinely fresh question for the next round
                try:
                    from llm import QuotaExhaustedError
                    next_question = _ask_opening_question(context, asked_questions)
                except QuotaExhaustedError as exc:
                    st.error(f"API Quota Exhausted: {exc}\n\nPlease try again later. Other non-AI features remain available.")
                    st.stop()
                asked_questions.append(next_question)
                _set("asked_questions", asked_questions)
                _set("current_question", next_question)
                messages.append({
                    "role": "assistant",
                    "content": next_question,
                    "score": None,
                    "breakdown": None,
                    "ideal_answer": None,
                })
                _set("messages", messages)
                _set("awaiting_answer", True)

            st.rerun()
