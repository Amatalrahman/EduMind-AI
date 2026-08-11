"""
Central configuration for AI Study Assistant.
All tunable knobs live here so nothing is hard-coded elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
BM25_DIR = DATA_DIR / "bm25_indices"
DB_PATH = DATA_DIR / "study_assistant.db"

# Create directories on import (safe to call multiple times)
for _d in [DATA_DIR, UPLOADS_DIR, CHROMA_DIR, BM25_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── Model Names ────────────────────────────────────────────────────────────────
GEMINI_TEXT_MODEL = "gemini-3.6-flash"
GEMINI_VISION_MODEL = "gemini-3.6-flash"          # same model supports vision
GROQ_MODEL = "llama-3.3-70b-versatile"

EMBED_MODEL = "all-MiniLM-L6-v2"                       # much smaller local sentence-transformer

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_TOKEN_LIMIT = 400                            # max tokens per chunk
CHUNK_OVERLAP_TOKENS = 50                          # overlap between consecutive chunks on same page
TIKTOKEN_ENCODING = "cl100k_base"                 # for token counting

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_DENSE = 10                                   # results from Chroma per query
TOP_K_BM25 = 10                                    # results from BM25 per query
TOP_K_FINAL = 6                                    # chunks fed to LLM after RRF
RRF_K = 60                                         # RRF constant

# ── Memory ─────────────────────────────────────────────────────────────────────
SHORT_TERM_TURNS = 5                               # number of prior chat turns kept as context

# ── Generation ─────────────────────────────────────────────────────────────────
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0.2
