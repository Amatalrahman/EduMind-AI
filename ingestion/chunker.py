"""
Page-aware text chunker.
Splits page text into chunks of at most CHUNK_TOKEN_LIMIT tokens using tiktoken.
NEVER merges text from different pages into the same chunk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import tiktoken

from config import CHUNK_TOKEN_LIMIT, CHUNK_OVERLAP_TOKENS, TIKTOKEN_ENCODING
from ingestion.pdf_parser import PageContent

logger = logging.getLogger(__name__)

# Load tokenizer once at module level
_enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""
    page_number: int
    chunk_index: int    # global index across the whole document
    text: str


def _tokenize(text: str) -> list[int]:
    return _enc.encode(text)


def _detokenize(tokens: list[int]) -> str:
    return _enc.decode(tokens)


def chunk_page(page_text: str, page_number: int, start_index: int = 0) -> list[Chunk]:
    """
    Split a single page's text into ≤CHUNK_TOKEN_LIMIT-token chunks with overlap.

    Args:
        page_text:    Raw text from one page.
        page_number:  1-indexed page / slide number.
        start_index:  Global chunk counter offset so indices remain unique per document.

    Returns:
        List of Chunk objects all labeled with page_number.
    """
    tokens = _tokenize(page_text)
    if not tokens:
        return []

    chunks: list[Chunk] = []
    pos = 0
    local_index = 0

    while pos < len(tokens):
        end = min(pos + CHUNK_TOKEN_LIMIT, len(tokens))
        chunk_tokens = tokens[pos:end]
        chunk_text = _detokenize(chunk_tokens).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    page_number=page_number,
                    chunk_index=start_index + local_index,
                    text=chunk_text,
                )
            )
            local_index += 1
        # advance with overlap so context isn't lost at boundaries
        step = CHUNK_TOKEN_LIMIT - CHUNK_OVERLAP_TOKENS
        pos += max(step, 1)

    return chunks


def chunk_pages(pages: list[PageContent]) -> list[Chunk]:
    """
    Chunk all pages in a document.
    Page boundaries are NEVER crossed — each chunk belongs to exactly one page.

    Args:
        pages: Output from pdf_parser.parse_pdf or pptx_parser.parse_pptx.

    Returns:
        Flat list of Chunks in document order.
    """
    all_chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        page_chunks = chunk_page(page.text, page.page_number, start_index=len(all_chunks))
        all_chunks.extend(page_chunks)

    logger.info(
        "Chunking complete: %d pages → %d chunks",
        len(pages),
        len(all_chunks),
    )
    return all_chunks
