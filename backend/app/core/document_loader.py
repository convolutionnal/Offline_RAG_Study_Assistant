"""
document_loader.py — PDF Document Loading Engine (pymupdf4llm)
================================================================

Responsible for discovering and extracting STRUCTURED MARKDOWN from
PDF files, returning Document objects whose page_content is Markdown
with preserved H1/H2/H3 headings, bold terms, and table formatting.

Primary parser: pymupdf4llm.to_markdown(page_chunks=True)
    - Converts each PDF page into Markdown preserving heading hierarchy,
      bold terms, tables, and code blocks.
    - The Markdown structure is critical: the downstream chunker uses
      MarkdownHeaderTextSplitter to split at topic boundaries.

Fallback parser: pdfplumber (for malformed table pages)
    - Called when pymupdf4llm produces unstructured output on table-heavy pages.

Quality gate: is_useful_page()
    - Filters out title pages, blank pages, and pure-number pages using
      MIN_TOKENS=40 and MIN_ALPHA_RATIO=0.4.

Usage:
    from app.core.document_loader import PDFDocumentLoader
    loader = PDFDocumentLoader(settings.UPLOAD_PATH)
    documents = loader.load_all()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Quality Gate Constants ───────────────────────────────────────────────
MIN_TOKENS = 40        # Pages with < 40 tokens are skipped (title pages, blank)
MIN_ALPHA_RATIO = 0.4  # At least 40% alphabetic chars (filters pure-number pages)


@dataclass
class Document:
    """
    A single page of extracted Markdown text with metadata.

    We define our own Document class instead of LangChain's to avoid
    coupling our pipeline to LangChain internals.

    Attributes:
        page_content: Markdown-formatted text from a single PDF page.
        metadata:     Dict with 'source', 'page', 'total_pages', 'file_path'.
    """
    page_content: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.page_content[:80].replace("\n", "\\n")
        return (
            f"Document(source='{self.metadata.get('source', '?')}', "
            f"page={self.metadata.get('page', '?')}, "
            f"chars={len(self.page_content)}, "
            f"preview='{preview}...')"
        )


def is_useful_page(page_text: str) -> bool:
    """
    Quality gate — filter out pages that would produce garbage chunks.

    A page is useful only if it has:
        - At least MIN_TOKENS words (filters title/blank pages)
        - At least MIN_ALPHA_RATIO fraction of alphabetic characters
          (filters pages that are just numbers/symbols)

    Args:
        page_text: Raw text content of a page.

    Returns:
        True if the page passes both quality checks.
    """
    tokens = len(page_text.split())
    alpha = sum(c.isalpha() for c in page_text)
    ratio = alpha / max(len(page_text), 1)
    return tokens >= MIN_TOKENS and ratio >= MIN_ALPHA_RATIO


def _extract_tables_fallback(pdf_path: str, page_num: int) -> str:
    """
    Fallback parser for table-heavy pages using pdfplumber.

    Called when pymupdf4llm produces malformed table Markdown.
    Extracts tables as clean Markdown tables.

    Args:
        pdf_path:  Path to the PDF file.
        page_num:  Zero-indexed page number to extract tables from.

    Returns:
        Markdown-formatted table string, or empty string if no tables found.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed — table fallback unavailable")
        return ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                return ""
            page = pdf.pages[page_num]
            tables = page.extract_tables()
            if not tables:
                return ""

            md_parts = []
            for table in tables:
                if not table or not table[0]:
                    continue
                # Build Markdown table from rows
                header = table[0]
                md_parts.append("| " + " | ".join(str(c or "") for c in header) + " |")
                md_parts.append("| " + " | ".join("---" for _ in header) + " |")
                for row in table[1:]:
                    md_parts.append("| " + " | ".join(str(c or "") for c in row) + " |")
                md_parts.append("")  # blank line after table

            return "\n".join(md_parts)
    except Exception as e:
        logger.warning(f"  pdfplumber fallback failed on page {page_num}: {e}")
        return ""


class PDFDocumentLoader:
    """
    Loads PDF files and extracts page-level MARKDOWN using pymupdf4llm.

    The Markdown output preserves heading hierarchy (H1/H2/H3), bold terms,
    tables, and code blocks — which the downstream MarkdownHeaderTextSplitter
    relies on for semantically-aware chunking.

    Args:
        pdf_directory: Path to the directory containing PDF files.
    """
    _SUPPORTED_EXTENSIONS: set[str] = {".pdf"}

    def __init__(self, pdf_directory: Path) -> None:
        self._pdf_dir = Path(pdf_directory)
        if not self._pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {self._pdf_dir}")
        if not self._pdf_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {self._pdf_dir}")
        logger.info(f"PDFDocumentLoader initialized — scanning [bold]{self._pdf_dir}[/bold]")

    def discover_pdfs(self) -> list[Path]:
        """Find all PDF files in the configured directory (sorted)."""
        pdf_files = sorted([
            f for f in self._pdf_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self._SUPPORTED_EXTENSIONS
        ])
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in: {self._pdf_dir}\n"
                f"Please add at least one .pdf file."
            )
        logger.info(f"Discovered [bold cyan]{len(pdf_files)}[/bold cyan] PDF(s)")
        return pdf_files

    def load_single(self, pdf_path: Path) -> list[Document]:
        """
        Extract Markdown from a single PDF using pymupdf4llm.

        Returns one Document per useful page. Pages that fail the
        quality gate (too short or non-textual) are skipped.
        Falls back to pdfplumber for pages with malformed tables.

        Returns empty list on failure (never crashes the pipeline).
        """
        try:
            import pymupdf4llm
        except ImportError:
            logger.error("pymupdf4llm not installed. Run: pip install pymupdf4llm")
            return []

        documents: list[Document] = []
        pdf_path = Path(pdf_path)
        source_name = pdf_path.name

        try:
            # -- Primary: pymupdf4llm -> structured Markdown -----------
            md_pages = pymupdf4llm.to_markdown(
                str(pdf_path),
                page_chunks=True,    # One dict per page
                show_progress=False,
            )

            total_pages = len(md_pages)
            logger.info(f"Loading [bold green]{source_name}[/bold green] ({total_pages} pages)")

            skipped_quality = 0
            skipped_empty = 0

            for page_data in md_pages:
                # pymupdf4llm returns list of dicts: {"text": str, "metadata": {...}}
                text = page_data.get("text", "")
                meta = page_data.get("metadata", {})
                page_num = meta.get("page", 0)

                # Skip completely empty pages
                if not text or not text.strip():
                    skipped_empty += 1
                    continue

                # ── Quality gate ─────────────────────────────────────
                if not is_useful_page(text):
                    skipped_quality += 1
                    logger.debug(
                        f"  Page {page_num + 1}/{total_pages} of {source_name} "
                        f"— failed quality gate (skipped)"
                    )
                    continue

                # ── Table fallback check ─────────────────────────────
                # If the Markdown has very few headings but the page is
                # likely table-heavy, supplement with pdfplumber output
                if "|" not in text and self._page_likely_has_tables(text):
                    table_md = _extract_tables_fallback(str(pdf_path), page_num)
                    if table_md:
                        text = text + "\n\n" + table_md
                        logger.debug(
                            f"  Page {page_num + 1}: supplemented with "
                            f"pdfplumber table extraction"
                        )

                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "total_pages": total_pages,
                        "file_path": str(pdf_path),
                    },
                ))

            total_chars = sum(len(d.page_content) for d in documents)
            logger.info(
                f"  Extracted {len(documents)}/{total_pages} pages, "
                f"{total_chars:,} chars from {source_name} "
                f"(skipped: {skipped_empty} empty, {skipped_quality} low-quality)"
            )

        except Exception as e:
            logger.error(f"  Failed to load {source_name}: {type(e).__name__}: {e}")
            return []

        return documents

    @staticmethod
    def _page_likely_has_tables(text: str) -> bool:
        """
        Heuristic: a page likely has tables if it contains many short
        lines with numeric content but no Markdown table pipes.
        """
        lines = text.strip().split("\n")
        if len(lines) < 3:
            return False
        short_numeric_lines = sum(
            1 for line in lines
            if len(line.strip()) < 60 and any(c.isdigit() for c in line)
        )
        return short_numeric_lines > len(lines) * 0.4

    def load_all(self) -> list[Document]:
        """
        Load ALL PDFs from the configured directory.

        Primary entry point for the ingestion pipeline. Returns a flat
        list of Documents across all files.
        """
        pdf_files = self.discover_pdfs()
        all_documents: list[Document] = []
        for pdf_path in pdf_files:
            all_documents.extend(self.load_single(pdf_path))

        if all_documents:
            sources = {d.metadata["source"] for d in all_documents}
            chars = sum(len(d.page_content) for d in all_documents)
            logger.info(f"[bold]Loading complete[/bold]: {len(all_documents)} pages from {len(sources)} PDF(s), {chars:,} chars")
        else:
            logger.warning("No text extracted from any PDF.")
        return all_documents

    def load_from_path(self, pdf_path: Path) -> list[Document]:
        """Load a single PDF from an arbitrary path (for Phase 2 uploads)."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() not in self._SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {pdf_path.suffix}")
        return self.load_single(pdf_path)
