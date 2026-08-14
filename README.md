

<p align="center">
  <img src="assets/logo.png" alt="EduMind logo" width="360"/>
</p>

<h3 align="center">Turn lecture slides and PDFs into an interactive study partner.</h3>

<p align="center">
  Upload your course material once. Ask it questions, summarize it, quiz yourself on it,
  drill it with flashcards, and teach it back — all grounded in your own documents.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Gemini-Q%26A%20%2B%20Vision-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Gemini"/>
  <img src="https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-F55036?style=for-the-badge&logo=groq&logoColor=white" alt="Groq"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-4B8BBE?style=for-the-badge" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/SQLite-Persistence-07405E?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite"/>
</p>

<p align="center">
  <sub>Final graduation project — <b>NTI (National Telecommunication Institute)</b> AI internship program</sub>
  <img width="960" height="464" alt="image" src="https://github.com/user-attachments/assets/08bf86e1-550f-49d0-ac70-7b4ac3fc6a44" />

</p>

<p align="center">
  <b>📄 Document Ingestion</b> &nbsp;·&nbsp;
  <b>💬 Grounded Chat Q&A</b> &nbsp;·&nbsp;
  <b>📝 Smart Summaries</b> &nbsp;·&nbsp;
  <b>🧠 Auto-Generated Quizzes</b> &nbsp;·&nbsp;
  <b>🃏 Flashcards</b> &nbsp;·&nbsp;
  <b>🎓 Explain It to Me</b>
</p>

---

## screenshot 

<p align="center">
<img width="1600" height="730" alt="6404d59b-63ec-4c6a-bf81-c76633a83c79" src="https://github.com/user-attachments/assets/daa7ddbc-1bde-4e56-a8fe-360d1f2c0cf8" />
<img width="1600" height="730" alt="618ea9fd-25f8-42fa-8c69-08d059b14ee4" src="https://github.com/user-attachments/assets/6b7b7fcc-de24-4df7-9a3e-90e53a66d548" />
<img width="1600" height="730" alt="b1bf6932-026e-4e39-b5aa-624e4d9d2357" src="https://github.com/user-attachments/assets/b9d60c9b-4894-46e5-a9a8-b1fd0325e188" />

  <img src="assets/demo.gif" alt="EduMind demo" width="800"/>
</p>

<p align="center">
  <sub>Replace <code>assets/demo.gif</code> with a screen recording or GIF of the app in action.</sub>
</p>

---

## 🚀 What EduMind Does

EduMind is a local-first, retrieval-augmented study assistant. Instead of chatting with a
generic model, every answer, quiz question, summary, and flashcard is generated from the
lectures you actually uploaded — with page-level citations, so you always know where an
answer came from.

---

## ✨ Features

### 📄 Document Ingestion
Drop in PDFs and PowerPoint decks. Text is extracted page by page, chunked, embedded, and
indexed automatically — no manual setup, no separate pipeline to run.

### 💬 Chat Q&A
Ask natural-language questions and get answers grounded strictly in your own notes.
Every claim is cited inline with the exact lecture and page number it came from.

### 📝 Summaries
Turn a dense chapter or an entire slide deck into a clear, structured summary — headings,
bullet points, and the key concepts pulled out for you.

### 🧠 Quizzes
Auto-generated multiple-choice quizzes built from your material, complete with
explanations for every answer and instant scoring.

### 🃏 Flashcards
AI-generated question-and-answer flashcards with a fast, distraction-free flip-to-review
interface — built for repetition, not clutter.

### 🎓 Explain It to Me
The standout feature. The AI plays a curious beginner and asks you to teach it a topic.
Your explanation gets scored across four dimensions — accuracy, completeness, clarity,
and depth — with a follow-up question that targets exactly what you missed.

---

## 🌟 Why It's Different

- **🎯 Grounded, not generic.** Every generated artifact — chat answer, quiz, flashcard, or
  summary — is built from a hybrid retrieval pipeline over your own documents, not the
  model's general knowledge, and the system prompt explicitly forbids answering outside
  the retrieved context.
- **🔀 Hybrid retrieval done right.** Dense vector search and sparse keyword search are fused
  with Reciprocal Rank Fusion, so both semantic meaning and exact terminology get
  surfaced — a single method alone would miss one or the other.
- **🧩 Learning science built in.** "Explain It to Me" flips the usual Q&A model — active
  recall and self-explanation are consistently shown to build deeper retention than
  passive review.
- **⚡ Fast where it counts.** Groq's LLaMA 3.3 70B handles latency-sensitive generation
  (quizzes, flashcards) while Gemini handles the reasoning-heavy Q&A and summarization
  work.
- **🔒 Your data stays yours.** Documents, embeddings, and progress are stored locally in
  SQLite and ChromaDB. Nothing leaves your machine beyond the LLM API calls you
  configure.

---

## ⚙️ How It Works

This is the actual pipeline running under the hood, end to end.

### 1️⃣ Ingestion

`ingestion/pdf_parser.py` and `ingestion/pptx_parser.py` open the uploaded file and walk
it page by page (or slide by slide), pulling out the raw text of each page separately.
Page boundaries are preserved deliberately — this is what later lets every citation point
to an exact page number instead of a vague "somewhere in this file."

### 2️⃣ Chunking

Handled by `ingestion/chunker.py`. Large pages of lecture text can't be fed to an LLM or
embedding model as-is, so each page is split into overlapping chunks using `tiktoken`:

- Each chunk is capped at `CHUNK_TOKEN_LIMIT` tokens (400 by default) — small enough to
  embed cleanly and stay within LLM context limits, large enough to preserve meaning.
- Consecutive chunks on the same page overlap by `CHUNK_OVERLAP_TOKENS` tokens (50 by
  default), so a sentence that happens to fall on a chunk boundary isn't cut in half and
  losing context.
- **Chunks never cross page boundaries.** Even if a page has very little text, it stays in
  its own chunk. This is what keeps citations page-accurate — every chunk is unambiguously
  "this page, this position."
- Each chunk keeps a `page_number` and a document-wide `chunk_index`, so it can always be
  traced back to its exact origin.

### 3️⃣ Embedding

`indexing/embedder.py` turns each chunk's text into a dense vector using a local
Sentence-Transformers model (no external API call, so this step is free and private).
Embeddings are L2-normalized, which makes cosine similarity search — used later by
ChromaDB — a simple dot product.

### 4️⃣ Indexing

Every chunk is written to two parallel indexes, one per subject:

- **ChromaDB** (`indexing/vector_store.py`) — a persistent vector collection per subject,
  storing embeddings plus metadata (filename, page number, chunk index) for dense
  semantic search.
- **BM25** (`indexing/bm25_index.py`) — a classic sparse keyword index built with
  `rank_bm25`, rebuilt per subject after every upload and pickled to disk. This is what
  catches exact terms, acronyms, and formulas that a purely semantic search can miss.

### 5️⃣ Hybrid Retrieval

When you ask a question, generate a quiz, or request a flashcard, `retrieval/hybrid_retriever.py`
runs both searches and fuses them with **Reciprocal Rank Fusion** (`retrieval/rrf.py`):

```
score(chunk) = Σ  1 / (k + rank_i(chunk))
```

For every ranked list a chunk appears in (dense, BM25), its rank contributes
`1 / (k + rank)` to its final score, with `k = 60` as in the original RRF paper. Chunks
that rank highly in *both* lists rise to the top; chunks that only one method found still
get a fair chance instead of being discarded. The top-scoring chunks are then enriched
with their source metadata from SQLite and returned.

### 6️⃣ Generation

`rag/qa_chain.py` builds the final prompt: retrieved chunks are formatted with explicit
`[Lecture: filename, p.X]` labels, and the system prompt instructs Gemini to answer
**only** from that context and cite every claim in that exact format. If nothing relevant
was retrieved, the system says so instead of guessing.

Quizzes and flashcards (`ui/tab_quizzes.py`, `ui/tab_flashcards.py`) follow the same
retrieve-then-generate pattern, but hand the context to Groq's LLaMA 3.3 70B instead —
chosen for its low latency, since generating a multi-question quiz needs to feel
near-instant.

### 7️⃣ Explain It to Me

`ui/reverse_teaching.py` inverts the usual flow: Gemini asks a beginner-level question
about the topic (grounded in the same retrieved context), you type an explanation in your
own words, and Gemini scores it across four dimensions — accuracy, completeness, clarity,
and depth — before asking a follow-up that probes whatever was missing. This forces active
recall instead of passive re-reading, which research consistently shows retains better.

---

## 🏗️ Architecture

```
                 ┌────────────────┐
                 │   Streamlit UI  │
                 └───────┬────────┘
                         │
        ┌────────────────┼─────────────────┐
        │                │                 │
┌───────▼───────┐ ┌──────▼───────┐ ┌───────▼────────┐
│   Ingestion    │ │   Indexing    │ │    Retrieval    │
│ PDF / PPTX      │ │ Embeddings    │ │ Dense + BM25    │
│ page-aware      │ │ ChromaDB      │ │ RRF fusion      │
│ chunking        │ │ + BM25 build  │ │                 │
└───────┬────────┘ └──────┬───────┘ └───────┬────────┘
        │                 │                 │
        └────────────────┼─────────────────┘
                         │
                 ┌───────▼────────┐
                 │   LLM Layer     │
                 │ Gemini + Groq   │
                 └───────┬────────┘
                         │
                 ┌───────▼────────┐
                 │   SQLite DB     │
                 │ subjects, docs, │
                 │ quizzes, cards  │
                 └────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Interface | Streamlit | Multi-page app shell and all interactive UI |
| Reasoning & vision | Google Gemini | Chat Q&A, summarization, reverse-teaching evaluation |
| Fast generation | Groq (LLaMA 3.3 70B) | Quiz and flashcard generation |
| Embeddings | Sentence-Transformers | Local, offline dense vector embeddings |
| Vector store | ChromaDB | Per-subject persistent semantic search |
| Keyword search | BM25 (rank-bm25) | Sparse lexical search, complements dense search |
| Result fusion | Reciprocal Rank Fusion | Merges dense + sparse rankings into one |
| Document parsing | PyMuPDF, python-pptx | Page-aware text extraction from PDF and PPTX |
| Persistence | SQLite | Subjects, documents, chunks, quiz results, flashcards |

---

## 📁 Project Structure

```
study_assistant/
├── app.py                  # Streamlit entry point and page routing
├── config.py                # Central configuration (models, chunking, retrieval)
├── ui/                      # One module per tab (upload, chat, quizzes, flashcards, ...)
├── ingestion/                # PDF / PPTX parsing and page-aware chunking
├── indexing/                 # Embedding generation, ChromaDB + BM25 index writes
├── retrieval/                 # Hybrid dense + BM25 retrieval with RRF
├── rag/                      # Question-answering and summarization chain
├── llm/                      # Gemini and Groq client wrappers
├── db/                       # SQLite schema and data access layer
└── assets/                   # Logo and UI imagery
```

---

## 🏁 Getting Started

### ✅ Prerequisites

- Python 3.10 or later
- A [Gemini API key](https://aistudio.google.com/app/apikey)
- A [Groq API key](https://console.groq.com/keys)

### 📦 Installation

```bash
git clone <repository-url>
cd study_assistant
pip install -r requirements.txt
```

### 🔑 Configuration

```bash
cp .env.example .env
```

Then fill in your keys:

```
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### ▶️ Run

```bash
streamlit run app.py
```

The app opens in your browser. Upload a PDF or PPTX from the **Upload** tab to get
started — every other tab pulls from what you've indexed.

---

## 🗺️ Roadmap

- 📅 Spaced-repetition scheduling that ties quiz, flashcard, and explanation performance
  together into a single review plan
- 🔗 Multi-document cross-referencing for broader Q&A
- 📤 Export of quizzes and flashcards to shareable formats

---

## 🙏 Acknowledgments

Built as a graduation project for the **NTI (National Telecommunication Institute)**
AI/ML internship track.

---

## 📜 License

Add your license of choice here.
