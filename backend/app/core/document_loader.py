"""
document_loader.py — PDF Document Loading Engine
==================================================

Responsible for discovering and extracting text from PDF files,
returning structured Document objects with rich metadata.

Architecture:
    This module handles ONLY loading — no cleaning, no chunking.
    Raw extracted text is passed to text_cleaner.py for sanitization.

Why PyMuPDF (fitz) over PyPDF2:
    - 5-10x faster extraction on large academic PDFs.
    - Superior handling of complex layouts, tables, unicode.
    - More reliable page boundary detection.

Usage:
    from app.core.document_loader import PDFDocumentLoader
    loader = PDFDocumentLoader(settings.paths.pdf_input_dir)
    documents = loader.load_all()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """
    A single page of extracted text with metadata.

    We define our own Document class instead of LangChain's to avoid
    coupling our pipeline to LangChain internals.

    Attributes:
        page_content: Raw extracted text from a single PDF page.
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


class PDFDocumentLoader:
    """
    Loads PDF files from a directory and extracts page-level text
    using PyMuPDF (fitz).

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
        Extract text from a single PDF, page by page.

        Returns empty list on failure (never crashes the pipeline).
        Pages with no extractable text are skipped with a warning.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF not installed. Run: pip install pymupdf")
            return []

        documents: list[Document] = []
        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            source_name = pdf_path.name
            logger.info(f"Loading [bold green]{source_name}[/bold green] ({total_pages} pages)")

            for page_num in range(total_pages):
                text = doc[page_num].get_text("text")
                if not text or not text.strip():
                    logger.warning(f"  Page {page_num+1}/{total_pages} of {source_name} — no text (skipped)")
                    continue
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "total_pages": total_pages,
                        "file_path": str(pdf_path),
                    },
                ))
            doc.close()
            total_chars = sum(len(d.page_content) for d in documents)
            logger.info(f"  Extracted {len(documents)} pages, {total_chars:,} chars from {source_name}")

        except Exception as e:
            logger.error(f"  Failed to load {pdf_path.name}: {type(e).__name__}: {e}")
            return []

        return documents

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
