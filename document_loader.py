"""
document_loader.py
------------------
Loads one or many PDFs from a file path or directory.
Returns a flat list of page dicts:
    { "text": str, "page_num": int, "source": str, "total_pages": int }

Handles:
- Normal text PDFs (lecture notes, textbooks)
- Multi-column layouts  (uses block sorting to fix reading order)
- Scanned/image PDFs    (detects them and warns — OCR not in scope here)
- Corrupt / unreadable  (skips with a warning, never crashes the whole batch)
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _is_scanned(page: fitz.Page, threshold: int = 20) -> bool:
    """
    Heuristic: if extracted text is shorter than `threshold` chars
    but the page has images, it's probably a scanned page.
    """
    text = page.get_text("text").strip()
    has_images = len(page.get_images()) > 0
    return len(text) < threshold and has_images


def _extract_text_sorted(page: fitz.Page) -> str:
    """
    Extracts text from a page using block-level sorting.
    This fixes multi-column PDFs where naive extraction
    interleaves two columns.

    Strategy: get all text blocks, sort top-to-bottom then
    left-to-right within the same horizontal band.
    """
    blocks = page.get_text("blocks")  # returns list of (x0, y0, x1, y1, text, ...)

    # Sort: primary = top of block (y0), secondary = left of block (x0)
    # Band tolerance: blocks within 20px vertical of each other = same row
    BAND = 20
    blocks_sorted = sorted(blocks, key=lambda b: (round(b[1] / BAND), b[0]))

    text_parts = [b[4].strip() for b in blocks_sorted if b[4].strip()]
    return "\n".join(text_parts)


# ── main loader ───────────────────────────────────────────────────────────────

def load_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Load a single PDF. Returns a list of page dicts.
    Skips scanned pages with a warning (no OCR).
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected .pdf, got: {path.suffix}")

    pages = []

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        logger.warning(f"Could not open {path.name}: {e}")
        return []

    total = len(doc)

    for page_num, page in enumerate(doc, start=1):
        if _is_scanned(page):
            logger.warning(
                f"[{path.name}] Page {page_num} appears scanned (image-only). "
                f"Skipping — add OCR support if needed."
            )
            continue

        text = _extract_text_sorted(page)

        if not text.strip():
            logger.debug(f"[{path.name}] Page {page_num} is empty. Skipping.")
            continue

        pages.append({
            "text": text,
            "page_num": page_num,
            "source": path.name,       # e.g. "os_notes.pdf"
            "total_pages": total,
        })

    doc.close()
    logger.info(f"Loaded {len(pages)}/{total} pages from {path.name}")
    return pages


def load_pdfs_from_dir(dir_path: str) -> List[Dict[str, Any]]:
    """
    Load ALL PDFs from a directory.
    Returns a flat list of page dicts across all files.
    Skips corrupt files with a warning — never crashes the whole batch.
    """
    dir_ = Path(dir_path)

    if not dir_.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    pdf_files = sorted(dir_.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {dir_path}")
        return []

    all_pages = []
    for pdf_file in pdf_files:
        try:
            pages = load_pdf(str(pdf_file))
            all_pages.extend(pages)
        except Exception as e:
            logger.warning(f"Skipping {pdf_file.name} due to error: {e}")

    logger.info(
        f"Loaded {len(all_pages)} total pages "
        f"from {len(pdf_files)} PDFs in {dir_path}"
    )
    return all_pages


def load_pdfs_from_list(pdf_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load a specific list of PDF paths.
    Useful when the API receives multiple uploaded files.
    """
    all_pages = []
    for path in pdf_paths:
        try:
            pages = load_pdf(path)
            all_pages.extend(pages)
        except Exception as e:
            logger.warning(f"Skipping {path} due to error: {e}")
    return all_pages
