"""
PDF parser using PyMuPDF (fitz).
Extracts text and images on a per-page basis.
Returns a list of page dicts so the chunker can process them page-by-page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
import pymupdf as fitz  # PyMuPDF (pymupdf >= 1.24 canonical import)
from ingestion.text_cleaner import clean_text

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Holds all content extracted from a single page / slide."""
    page_number: int          # 1-indexed
    text: str
    images: list[bytes] = field(default_factory=list)   # raw PNG bytes per image


def parse_pdf(file_path: str | Path) -> list[PageContent]:
    """
    Open a PDF and extract text + images from every page.

    Args:
        file_path: path to the PDF file on disk.

    Returns:
        List of PageContent objects, one per page.
    """
    file_path = Path(file_path)
    pages: list[PageContent] = []

    doc = fitz.open(str(file_path))
    logger.info("Parsing PDF: %s  (%d pages)", file_path.name, len(doc))

    for page_index, page in enumerate(doc):
        page_number = page_index + 1  # 1-indexed

        # ── Text ──────────────────────────────────────────────────────────────
        text = page.get_text("text")  # plain text extraction
        text = clean_text(text)

        # ── Images ────────────────────────────────────────────────────────────
        raw_images: list[bytes] = []
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            try:
                # Convert to PNG for consistency
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:  # CMYK or other – convert to RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                raw_images.append(pix.tobytes("png"))
                pix = None  # free memory
            except Exception as exc:
                logger.warning("Could not extract image xref=%d on page %d: %s", xref, page_number, exc)

        pages.append(PageContent(page_number=page_number, text=text, images=raw_images))

    doc.close()
    logger.info("Finished parsing PDF: %d pages extracted", len(pages))
    return pages
