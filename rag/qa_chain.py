"""
RAG Q&A chain.
Orchestrates: retrieve → build prompt → call Gemini → return structured answer.
Every answer includes citations in the form [Lecture: {filename}, p.{page}].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from config import SHORT_TERM_TURNS, TOP_K_FINAL
from retrieval.hybrid_retriever import retrieve, RetrievedChunk
from llm import gemini_client

logger = logging.getLogger(__name__)


# ── Prompt templates ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert study assistant helping a student understand their lecture materials.

Rules you MUST follow:
1. Answer ONLY from the provided context chunks. Do not hallucinate or use outside knowledge.
2. For every factual claim, cite its source inline using the format: [Lecture: <filename>, p.<page>]
3. If the context does not contain enough information to answer the question, say so clearly.
4. Be concise, structured, and educational. Use bullet points or numbered lists when appropriate.
5. If the question cannot be answered from context, suggest what topic the student might look up.
"""

CONTEXT_TEMPLATE = """--- CONTEXT CHUNK {idx} ---
Lecture: {filename}
Page: {page}
---
{text}
"""

QUERY_TEMPLATE = """Based on the context chunks above, please answer the following question:

{question}

Remember to cite every claim with [Lecture: <filename>, p.<page>].
"""


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ChatTurn:
    role: str   # "user" or "assistant"
    content: str


@dataclass
class RAGAnswer:
    answer: str
    sources: list[dict]          # [{"filename": ..., "page": ..., "text": ...}]
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        parts.append(
            CONTEXT_TEMPLATE.format(
                idx=idx,
                filename=chunk.filename,
                page=chunk.page_number,
                text=chunk.text,
            )
        )
    return "\n".join(parts)


def _build_history_for_gemini(history: list[ChatTurn]) -> list[dict]:
    """Convert ChatTurn list to Gemini history format."""
    gemini_history = []
    for turn in history:
        role = "user" if turn.role == "user" else "model"
        gemini_history.append({"role": role, "parts": [turn.content]})
    return gemini_history


def _extract_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    """Build a de-duplicated source list from retrieved chunks."""
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk.filename, chunk.page_number)
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": chunk.filename,
                "page": chunk.page_number,
                "snippet": chunk.text[:200] + "…" if len(chunk.text) > 200 else chunk.text,
            })
    return sources


# ── Main QA function ───────────────────────────────────────────────────────────

def answer_question(
    question: str,
    subject_id: int,
    chat_history: Optional[list[ChatTurn]] = None,
    top_k: int = TOP_K_FINAL,
) -> RAGAnswer:
    """
    Full RAG pipeline: retrieve → prompt → generate → return.

    Args:
        question:     The student's question.
        subject_id:   Which subject to retrieve from (covers all its lectures).
        chat_history: Prior conversation turns for short-term memory.
        top_k:        Number of chunks to feed to the LLM.

    Returns:
        RAGAnswer with the generated text and source metadata.
    """
    # 1. Retrieve
    chunks = retrieve(query=question, subject_id=subject_id, top_k=top_k)

    if not chunks:
        return RAGAnswer(
            answer=(
                "I couldn't find relevant information in your uploaded materials for this question. "
                "Please make sure you've uploaded and indexed documents for this subject."
            ),
            sources=[],
            retrieved_chunks=[],
        )

    # 2. Build prompt
    context_block = _format_context(chunks)
    full_prompt = context_block + "\n\n" + QUERY_TEMPLATE.format(question=question)

    # 3. Short-term memory: trim to last N turns
    history = chat_history or []
    recent_history = history[-(SHORT_TERM_TURNS * 2):]   # each turn = user + model msg
    gemini_history = _build_history_for_gemini(recent_history)

    # 4. Call Gemini
    try:
        answer_text = gemini_client.generate_text(
            prompt=full_prompt,
            system_instruction=SYSTEM_PROMPT,
            history=gemini_history if gemini_history else None,
        )
    except Exception as exc:
        logger.exception("Gemini API error during Q&A")
        return RAGAnswer(
            answer=f"❌ Gemini API error: {exc}",
            sources=[],
            retrieved_chunks=chunks,
        )

    # 5. Package result
    sources = _extract_sources(chunks)
    return RAGAnswer(
        answer=answer_text,
        sources=sources,
        retrieved_chunks=chunks,
    )


# ── Summarization helpers (used by Summaries tab) ─────────────────────────────

SUMMARIZE_SYSTEM = """You are an expert academic summarizer.
Create a structured, comprehensive summary of the provided lecture material.
Use headings, bullet points, and highlight key concepts, definitions, and examples.
"""

SUMMARIZE_PROMPT = """Summarize the following lecture content retrieved from '{title}'.
Produce a well-structured summary with clear sections.

{context}
"""


def summarize(
    subject_id: int,
    title: str,
    query: str = "summarize all main topics and key concepts",
    top_k: int = 12,
) -> str:
    """
    Generate a summary by retrieving the most representative chunks and asking Gemini to summarize them.
    Works for both per-lecture and per-subject summaries depending on the caller's query.
    """
    chunks = retrieve(query=query, subject_id=subject_id, top_k=top_k)
    if not chunks:
        return "No content found for summarization. Please upload documents first."

    context = _format_context(chunks)
    prompt = SUMMARIZE_PROMPT.format(title=title, context=context)

    try:
        return gemini_client.generate_text(
            prompt=prompt,
            system_instruction=SUMMARIZE_SYSTEM,
        )
    except Exception as exc:
        logger.exception("Gemini API error during summarization")
        return f"❌ Summarization failed: {exc}"
