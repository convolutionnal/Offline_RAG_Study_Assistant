"""
text_cleaner.py — Multi-Stage Text Cleaning Pipeline
======================================================

Cleans raw PDF-extracted text to remove noise without losing
contextual integrity. Each cleaning step is a separate private
method that can be individually toggled or reordered.

Cleaning Pipeline (applied in order):
    1. Fix broken hyphenation    - "algo-\\nrithm" -> "algorithm"
    2. Remove control characters — Strip \\x00-\\x1f except \\n, \\t
    3. Detect & strip repeating headers/footers across pages
    4. Strip standalone page numbers
    5. Normalize whitespace       — Collapse excessive newlines/spaces

Why this order matters:
    - Hyphenation must be fixed BEFORE whitespace normalization,
      otherwise the linebreak between "algo-" and "rithm" gets
      collapsed and the hyphen remains orphaned.
    - Headers/footers must be detected BEFORE page numbers are
      stripped, because page numbers are often part of header lines.

Usage:
    from app.core.text_cleaner import TextCleaner
    from app.core.document_loader import Document

    cleaner = TextCleaner()
    cleaned_docs = cleaner.clean(raw_documents)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from app.core.document_loader import Document
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """
    A configurable, multi-stage text cleaning pipeline.

    Each stage is a separate method, so individual steps can be
    disabled via constructor flags if a specific PDF format
    requires it (e.g., keeping headers for legal documents).

    Args:
        fix_hyphenation:    Rejoin words broken across line endings.
        remove_headers:     Detect and remove repeating headers/footers.
        strip_page_numbers: Remove standalone page number lines.
        normalize_spaces:   Collapse excessive whitespace.
        header_threshold:   Fraction of pages a line must appear in
                            to be considered a repeating header (0.0-1.0).
    """

    def __init__(
        self,
        fix_hyphenation: bool = True,
        remove_headers: bool = True,
        strip_page_numbers: bool = True,
        normalize_spaces: bool = True,
        header_threshold: float = 0.5,
    ) -> None:
        self._fix_hyphenation = fix_hyphenation
        self._remove_headers = remove_headers
        self._strip_page_numbers = strip_page_numbers
        self._normalize_spaces = normalize_spaces
        self._header_threshold = header_threshold

        logger.info(
            f"TextCleaner initialized — "
            f"hyphen={fix_hyphenation}, headers={remove_headers}, "
            f"page_nums={strip_page_numbers}, normalize={normalize_spaces}"
        )

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def clean(self, documents: list[Document]) -> list[Document]:
        """
        Run the full cleaning pipeline on a list of Documents.

        Each Document's page_content is cleaned in-place (a new
        Document object is created to preserve immutability of the
        original). Metadata is preserved unchanged.

        Args:
            documents: List of raw Document objects from the loader.

        Returns:
            New list of Document objects with cleaned page_content.
            Empty documents (all content was noise) are filtered out.
        """
        if not documents:
            logger.warning("No documents to clean.")
            return []

        logger.info(f"Cleaning {len(documents)} document pages...")

        # ── Step 1: Detect repeating headers/footers ─────────────────
        # This must happen BEFORE per-page cleaning because it needs
        # to compare lines across ALL pages to find repetitions.
        headers_to_remove: set[str] = set()
        if self._remove_headers:
            headers_to_remove = self._detect_repeating_lines(documents)

        # ── Step 2: Clean each page ──────────────────────────────────
        cleaned_docs: list[Document] = []
        total_chars_before = 0
        total_chars_after = 0

        for doc in documents:
            original_text = doc.page_content
            total_chars_before += len(original_text)

            cleaned_text = original_text

            # Apply cleaning stages in order
            if self._fix_hyphenation:
                cleaned_text = self._rejoin_hyphenated_words(cleaned_text)

            cleaned_text = self._remove_control_characters(cleaned_text)

            if self._remove_headers and headers_to_remove:
                cleaned_text = self._strip_known_headers(
                    cleaned_text, headers_to_remove
                )

            if self._strip_page_numbers:
                cleaned_text = self._strip_page_number_lines(cleaned_text)

            if self._normalize_spaces:
                cleaned_text = self._normalize_whitespace(cleaned_text)

            # Skip pages that became empty after cleaning
            if not cleaned_text.strip():
                logger.debug(
                    f"  Page {doc.metadata.get('page', '?')} of "
                    f"{doc.metadata.get('source', '?')} — "
                    f"empty after cleaning (dropped)"
                )
                continue

            total_chars_after += len(cleaned_text)

            cleaned_docs.append(Document(
                page_content=cleaned_text,
                metadata=doc.metadata.copy(),  # Preserve original metadata
            ))

        # ── Summary ──────────────────────────────────────────────────
        reduction = (
            (1 - total_chars_after / total_chars_before) * 100
            if total_chars_before > 0 else 0
        )
        logger.info(
            f"[bold]Cleaning complete[/bold]: "
            f"{len(cleaned_docs)}/{len(documents)} pages retained, "
            f"{total_chars_before:,} -> {total_chars_after:,} chars "
            f"({reduction:.1f}% noise removed)"
        )

        return cleaned_docs

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string (no metadata handling).

        Useful for cleaning individual chunks or query text
        before embedding.

        Args:
            text: Raw text string to clean.

        Returns:
            Cleaned text string.
        """
        result = text
        if self._fix_hyphenation:
            result = self._rejoin_hyphenated_words(result)
        result = self._remove_control_characters(result)
        if self._strip_page_numbers:
            result = self._strip_page_number_lines(result)
        if self._normalize_spaces:
            result = self._normalize_whitespace(result)
        return result.strip()

    # ══════════════════════════════════════════════════════════════════
    # Private Cleaning Stages
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _rejoin_hyphenated_words(text: str) -> str:
        """
        Fix words broken across line endings by a hyphen.

        Pattern: 'algo-\\nrithm' -> 'algorithm'

        The regex matches:
            - A word character before the hyphen (to avoid list items)
            - A hyphen at the end of a line
            - Optional whitespace + newline
            - A lowercase letter starting the next line

        We only rejoin when the next line starts lowercase, to avoid
        breaking intentional hyphenation like "well-known".
        """
        return re.sub(
            r"(\w)-\s*\n\s*([a-z])",
            r"\1\2",
            text,
        )

    @staticmethod
    def _remove_control_characters(text: str) -> str:
        """
        Strip ASCII control characters (0x00-0x1F) except:
            - \\n (newline, 0x0A) — needed for structure
            - \\t (tab, 0x09)    — used in some table layouts

        These control chars sometimes leak from PDF binary streams
        and cause encoding issues downstream.
        """
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    def _detect_repeating_lines(
        self, documents: list[Document]
    ) -> set[str]:
        """
        Identify lines that appear across many pages — likely
        headers, footers, or watermarks.

        Algorithm:
            1. For each page, extract the first 3 and last 3 lines.
            2. Normalize each line (strip whitespace, lowercase).
            3. Count how many pages each normalized line appears in.
            4. Lines appearing in > threshold fraction of pages are
               flagged as repeating headers/footers.

        We only check the first/last 3 lines because headers and
        footers are always at page boundaries, never mid-page.

        Args:
            documents: All page-level Documents from the loader.

        Returns:
            Set of normalized line strings to remove.
        """
        total_pages = len(documents)
        if total_pages < 3:
            # Need at least 3 pages to reliably detect repetition.
            return set()

        line_page_counts: Counter[str] = Counter()

        for doc in documents:
            lines = doc.page_content.strip().split("\n")
            lines = [ln.strip() for ln in lines if ln.strip()]

            # Extract boundary lines (first 3 + last 3)
            boundary_lines: set[str] = set()
            for line in lines[:3]:
                normalized = line.strip().lower()
                if len(normalized) > 3:  # Skip very short lines
                    boundary_lines.add(normalized)
            for line in lines[-3:]:
                normalized = line.strip().lower()
                if len(normalized) > 3:
                    boundary_lines.add(normalized)

            # Count each unique line once per page
            for norm_line in boundary_lines:
                line_page_counts[norm_line] += 1

        # Flag lines appearing in > threshold fraction of pages
        min_occurrences = int(total_pages * self._header_threshold)
        repeating = {
            line for line, count in line_page_counts.items()
            if count >= min_occurrences
        }

        if repeating:
            logger.info(
                f"  Detected {len(repeating)} repeating header/footer pattern(s)"
            )
            for line in list(repeating)[:5]:  # Show first 5
                logger.debug(f"    -> '{line[:60]}...'")

        return repeating

    @staticmethod
    def _strip_known_headers(text: str, headers: set[str]) -> str:
        """
        Remove lines matching known repeating headers/footers.

        Comparison is case-insensitive with whitespace stripped
        to handle minor formatting variations across pages.
        """
        lines = text.split("\n")
        filtered = [
            line for line in lines
            if line.strip().lower() not in headers
        ]
        return "\n".join(filtered)

    @staticmethod
    def _strip_page_number_lines(text: str) -> str:
        """
        Remove lines that are likely standalone page numbers.

        Matches lines containing ONLY:
            - Digits (e.g., '42')
            - Digits with surrounding dashes/dots (e.g., '- 42 -', '..42..')
            - Common page patterns: 'Page 42', 'p. 42', '42 of 100'

        Does NOT remove lines where numbers are part of content
        (e.g., 'Chapter 3: Memory Management').
        """
        page_number_pattern = re.compile(
            r"^\s*"
            r"(?:"
            r"[-–—.\s]*\d{1,4}[-–—.\s]*"   # Standalone number with optional dashes/dots
            r"|page\s*\d{1,4}"               # 'Page 42'
            r"|p\.\s*\d{1,4}"               # 'p. 42'
            r"|\d{1,4}\s*of\s*\d{1,4}"      # '42 of 100'
            r")"
            r"\s*$",
            re.IGNORECASE,
        )
        lines = text.split("\n")
        filtered = [
            line for line in lines
            if not page_number_pattern.match(line)
        ]
        return "\n".join(filtered)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """
        Collapse excessive whitespace while preserving structure.

        Steps:
            1. Strip trailing whitespace from each line.
            2. Collapse 3+ consecutive newlines into exactly 2
               (preserves paragraph breaks but removes page gaps).
            3. Collapse multiple spaces into one (within lines).
            4. Strip leading/trailing whitespace from the full text.
        """
        # Strip trailing spaces per line
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

        # Collapse 3+ newlines -> 2 (keep paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces -> single space (within lines)
        text = re.sub(r" {2,}", " ", text)

        return text.strip()
