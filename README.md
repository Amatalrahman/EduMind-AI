# 🎓 EduMind AI - AI Study Assistant

An intelligent AI study assistant built with **LangChain**, **RAG (Retrieval-Augmented Generation)**, and **LLMs** to help students interact with their study materials, summarize text, and generate practice quizzes.

---

## 📌 Features

- **📄 Document Processing & Chunking:** Preprocess and clean Arabic & English PDF study materials while keeping metadata.
- **🔍 Smart Retrieval (RAG):** Ask questions about lectures and receive accurate answers with cited pages.
- **📝 Automatic Summarization:** Get concise summaries and key takeaways from chapters.
- **❓ Quiz Generation:** Generate multiple-choice questions (MCQs) for self-assessment.

---

## 🛠️ Tech Stack

- **Framework:** LangChain
- **PDF Processing:** PyPDF & PyMuPDF
- **Text Splitting:** RecursiveCharacterTextSplitter
- **Vector Database:** ChromaDB / FAISS
- **Language Models:** Multilingual Embeddings & LLMs (Gemini / OpenAI)
- **User Interface:** Streamlit / Gradio

---

## 📂 Project Structure

```text
EduMind-AI/
├── data/               # Raw study materials (PDFs)
├── notebooks/          # Colab Notebooks for preprocessing & testing
├── README.md           # Project documentation
└── requirements.txt    # Required python packages
