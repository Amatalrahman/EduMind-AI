"""
Basic text cleaning for extracted PDF and PPTX content.
Preserves Arabic and English text and paragraph structure.
"""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """
    Clean extracted text while preserving its original meaning.
    """

    # Remove null characters
    text = text.replace("\x00", " ")
    # Normalize excessive spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Keep paragraph structure
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,!?;:؟،])", r"\1", text)

    return text.strip()