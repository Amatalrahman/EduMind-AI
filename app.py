"""
AI Study Assistant — Streamlit application entry point.
Run with: streamlit run app.py
"""

import logging
import sys
from pathlib import Path

import streamlit as st

# ── Path setup (so sibling packages resolve) ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bootstrap DB ──────────────────────────────────────────────────────────────
from db.database import init_db
init_db()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Tab imports ───────────────────────────────────────────────────────────────
from ui.tab_upload import render_upload_tab
from ui.tab_chat import render_chat_tab
from ui.tab_summaries import render_summaries_tab
from ui.tab_quizzes import render_quizzes_tab
from ui.tab_flashcards import render_flashcards_tab
from ui.tab_study_plan import render_study_plan_tab

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🎓 AI Study Assistant")
    st.markdown(
        "Your personal AI-powered study companion. "
        "Upload lectures, ask questions, generate quizzes, and track your progress."
    )
    st.divider()
    st.markdown("**Powered by**")
    st.markdown("- 🤖 Gemini-3.6-flash (Q&A + Vision)")
    st.markdown("- ⚡ Groq LLaMA 3.3 70B (Quizzes)")
    st.markdown("- 🔍 BAAI/bge-m3 (Embeddings)")
    st.markdown("- 🗄️ ChromaDB + BM25 (Hybrid RAG)")
    st.divider()
    st.caption("All data stored locally. No cloud required.")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_upload, tab_chat, tab_summaries, tab_quizzes, tab_flashcards, tab_study_plan = st.tabs([
    "📂 Upload",
    "💬 Chat / Q&A",
    "📝 Summaries",
    "🧠 Quizzes",
    "🃏 Flashcards",
    "📅 Study Plan",
])

with tab_upload:
    render_upload_tab()

with tab_chat:
    render_chat_tab()

with tab_summaries:
    render_summaries_tab()

with tab_quizzes:
    render_quizzes_tab()

with tab_flashcards:
    render_flashcards_tab()

with tab_study_plan:
    render_study_plan_tab()
