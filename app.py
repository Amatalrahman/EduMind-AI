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
from ui.reverse_teaching import render_reverse_teaching_tab

# ── Styles ────────────────────────────────────────────────────────────────────
from ui.styles import inject_custom_css
inject_custom_css()

# ── Main UI Routing ───────────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

def navigate(page: str):
    st.session_state.current_page = page

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🧠 EduMind")
    st.markdown(
        "Your personal AI-powered study companion. "
        "Upload lectures, ask questions, generate quizzes, and track your progress."
    )
    if st.session_state.current_page != "Home":
        if st.button("🏠 Home Dashboard", use_container_width=True):
            st.session_state.current_page = "Home"
            st.rerun()

    st.divider()
    st.markdown("**Powered by**")
    st.markdown("- 🤖 Gemini 2.0 Flash (Q&A + Vision)")
    st.markdown("- ⚡ Groq LLaMA 3.3 70B (Quizzes)")
    st.markdown("- 🔍 BAAI/bge-m3 (Embeddings)")
    st.markdown("- 🗄️ ChromaDB + BM25 (Hybrid RAG)")
    st.divider()
    st.caption("All data stored locally. No cloud required.")

# ── Views ─────────────────────────────────────────────────────────────────────
if st.session_state.current_page == "Home":
    st.markdown("<h1 style='text-align: center;'>Welcome to EduMind</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>What would you like to do today?</p>", unsafe_allow_html=True)
    st.write("")
    st.write("")

    features = [
        {"id": "upload", "title": "Upload", "desc": "Add new lectures", "icon": "upload.jpg"},
        {"id": "chat", "title": "Chat Q&A", "desc": "Ask your notes", "icon": "chat.png"},
        {"id": "summaries", "title": "Summaries", "desc": "Quick digests", "icon": "summarization.jpg"},
        {"id": "quizzes", "title": "Quizzes", "desc": "Test yourself", "icon": "quizzes.jpg"},
        {"id": "flashcards", "title": "Flashcards", "desc": "Spaced repetition", "icon": "flash_cards.png"},
        {"id": "plan", "title": "Study Plan", "desc": "Your schedule", "icon": "plan.png"},
        {"id": "explain", "title": "Explain to Me", "desc": "Reverse teaching", "icon": "explain.png"},
    ]

    import os
    cards_dir = Path("assets/cards")
    cards_dir.mkdir(parents=True, exist_ok=True)

    # Render Grid
    for row_idx in range(0, 6, 3):
        cols = st.columns(3)
        for col_idx in range(3):
            f_idx = row_idx + col_idx
            if f_idx < len(features):
                feature = features[f_idx]
                with cols[col_idx]:
                    with st.container(border=True):
                        img_path = cards_dir / feature["icon"]
                        if img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.write(f"*(Missing {feature['icon']})*")
                        st.markdown(f"<h3 style='text-align: center; margin-bottom: 5px;'>{feature['title']}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p style='text-align: center; color: #666; font-size: 0.9em; margin-bottom: 15px;'>{feature['desc']}</p>", unsafe_allow_html=True)
                        st.button("Open", key=feature["id"], use_container_width=True, on_click=navigate, args=(feature["id"],))
        st.write("")

    # Center the last item (Explain to Me)
    _, c2, _ = st.columns([1, 1, 1])
    with c2:
        feature = features[6]
        with st.container(border=True):
            img_path = cards_dir / feature["icon"]
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.write(f"*(Missing {feature['icon']})*")
            st.markdown(f"<h3 style='text-align: center; margin-bottom: 5px;'>{feature['title']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: #666; font-size: 0.9em; margin-bottom: 15px;'>{feature['desc']}</p>", unsafe_allow_html=True)
            st.button("Open", key=feature["id"], use_container_width=True, on_click=navigate, args=(feature["id"],))

else:
    # Subpage view
    if st.button("⬅️ Back to Dashboard"):
        st.session_state.current_page = "Home"
        st.rerun()
    st.divider()

    page = st.session_state.current_page
    if page == "upload":
        render_upload_tab()
    elif page == "chat":
        render_chat_tab()
    elif page == "summaries":
        render_summaries_tab()
    elif page == "quizzes":
        render_quizzes_tab()
    elif page == "flashcards":
        render_flashcards_tab()
    elif page == "plan":
        render_study_plan_tab()
    elif page == "explain":
        render_reverse_teaching_tab()
