"""
chunker.py - Hybrid Header + Semantic Chunker
==================================================================

Two-pass chunking pipeline for academic PDFs:

  Pass 1 - Header-Aware Split (MarkdownHeaderTextSplitter):
      Splits Markdown text at H1/H2/H3 heading boundaries.
      Academic notes are organized by topic, so each section heading
      IS a natural semantic boundary. This is the single biggest
      quality improvement for retrieval.

  Pass 2 - Semantic Split (for oversized sections):
      If a header-section is too large (>max_chunk_size chars), the
      existing semantic chunker (embedding-based adaptive breakpoints)
      splits it further at natural topic shifts within the section.

  Post-processing:
      - Section metadata (breadcrumb) is prepended to each chunk text
        for embedding, e.g. "Chapter 3 > Scheduling > FCFS: ..."
      - Size guardrails enforce min/max chunk sizes.
      - Deterministic IDs enable idempotent storage.

Usage:
    from app.core.chunker import SemanticChunker
    from app.core.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    chunker = SemanticChunker(embedding_engine=engine)
    chunks = chunker.chunk(cleaned_documents)

Dependencies:
    - langchain-text-splitters (MarkdownHeaderTextSplitter)
    - sentence-transformers (via EmbeddingEngine)
    - nltk (punkt_tab tokenizer for sentence splitting)
    - numpy (cosine similarity computation)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from app.config import settings
from app.core.document_loader import Document
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.core.embeddings import EmbeddingEngine

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """
    A semantically coherent text chunk with rich metadata.

    This is the fundamental unit stored in ChromaDB and retrieved
    during query time. Each chunk represents a complete "thought unit"
    extracted from the source document.

    Attributes:
        text:       The chunk content (with breadcrumb prepended).
        chunk_id:   Deterministic ID for idempotent storage.
                    Computed as SHA-256 of (source + chunk_index).
        metadata:   Dictionary containing provenance information:
                    - source:        PDF filename
                    - page_start:    First page this chunk spans
                    - page_end:      Last page this chunk spans
                    - chunk_index:   Sequential index within the document
                    - num_sentences: How many sentences were merged
                    - char_count:    Character length of the chunk
                    - section:       H1 heading this chunk belongs to
                    - subsection:    H2 heading this chunk belongs to
                    - subsubsection: H3 heading this chunk belongs to
    """

    text: str
    chunk_id: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", "\\n")
        return (
            f"Chunk(id='{self.chunk_id[:12]}...', "
            f"source='{self.metadata.get('source', '?')}', "
            f"pages={self.metadata.get('page_start', '?')}-"
            f"{self.metadata.get('page_end', '?')}, "
            f"chars={len(self.text)}, "
            f"preview='{preview}...')"
        )


# ═══════════════════════════════════════════════════════════════════════
# Sentence Tokenizer (NLTK)
# ═══════════════════════════════════════════════════════════════════════

def _ensure_nltk_data() -> None:
    """
    Download the NLTK punkt_tab tokenizer if not already present.

    punkt_tab is the updated sentence tokenizer that replaces the
    legacy 'punkt' resource. It handles abbreviations, decimal
    numbers, and academic citation styles correctly.

    This runs once and caches to ~/nltk_data (fully offline afterward).
    """
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("Downloading NLTK punkt_tab tokenizer (one-time)...")
        nltk.download("punkt_tab", quiet=True)
        logger.info("NLTK punkt_tab downloaded successfully.")


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using NLTK's punkt_tab tokenizer.

    Why NLTK over regex:
        - Handles abbreviations: "Dr. Smith went..." -> 1 sentence
        - Handles decimals: "scored 3.5 points" -> not split at "3."
        - Handles ellipses: "wait... what?" -> 1 sentence
        - Academic-aware: "et al. (2020)" -> not split

    Args:
        text: Cleaned document text.

    Returns:
        List of sentence strings. Empty sentences are filtered out.
    """
    import nltk

    _ensure_nltk_data()

    sentences = nltk.sent_tokenize(text, language="english")

    # Filter out empty/whitespace-only sentences
    return [s.strip() for s in sentences if s.strip()]


# ================================================================
# Header-Aware Split (Pass 1) - MarkdownHeaderTextSplitter
# ================================================================

# Headers to split on - matches H1, H2, H3 in Markdown output
HEADERS_TO_SPLIT_ON = [
    ("#",   "section"),
    ("##",  "subsection"),
    ("###", "subsubsection"),
]


def _split_by_headers(markdown_text: str) -> list[dict]:
    """
    Split Markdown text at heading boundaries using LangChain's
    MarkdownHeaderTextSplitter.

    Each resulting chunk contains text from a single section,
    with metadata indicating which headings it belongs to.

    Args:
        markdown_text: Full Markdown text from pymupdf4llm.

    Returns:
        List of dicts with 'text' and 'metadata' (section, subsection, etc.).
        If no headers are found, returns the full text as a single chunk.
    """
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter
    except ImportError:
        logger.warning(
            "langchain-text-splitters not installed - "
            "falling back to full-text chunking"
        )
        return [{"text": markdown_text, "metadata": {}}]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # Keep headers in chunk text for context
    )

    docs = splitter.split_text(markdown_text)

    if not docs:
        # No headers found - return full text as single chunk
        return [{"text": markdown_text, "metadata": {}}]

    results = []
    for doc in docs:
        results.append({
            "text": doc.page_content,
            "metadata": dict(doc.metadata) if doc.metadata else {},
        })

    logger.debug(f"  Header split produced {len(results)} section(s)")
    return results


def _build_breadcrumb(metadata: dict) -> str:
    """
    Build a heading breadcrumb string from section metadata.

    Example: "Chapter 3 > Process Scheduling > FCFS Algorithm"

    This breadcrumb is prepended to chunk text before embedding,
    which dramatically improves retrieval specificity by encoding
    the section context into the embedding vector.
    """
    parts = []
    for key in ["section", "subsection", "subsubsection"]:
        if key in metadata and metadata[key]:
            parts.append(metadata[key])
    return " > ".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Core: Semantic Chunker
# ═══════════════════════════════════════════════════════════════════════

class SemanticChunker:
    """
    Splits documents into semantically coherent chunks by detecting
    natural topic boundaries using embedding similarity.

    The chunker is stateless — it does not cache results between
    calls. Each invocation of chunk() processes documents independently.

    Args:
        embedding_engine:   An initialized EmbeddingEngine instance
                            used to embed individual sentences.
        breakpoint_k:       Sensitivity multiplier for the breakpoint
                            threshold. Higher k -> fewer, larger chunks.
                            Default from config: 1.0
        min_chunk_size:     Minimum character count per chunk. Chunks
                            below this are merged with neighbors.
                            Default from config: 100
        max_chunk_size:     Maximum character count per chunk. Chunks
                            above this are split at sentence boundaries.
                            Default from config: 2000
    """

    def __init__(
        self,
        embedding_engine: "EmbeddingEngine",
        breakpoint_k: float | None = None,
        min_chunk_size: int | None = None,
        max_chunk_size: int | None = None,
    ) -> None:
        self._engine = embedding_engine
        self._k = breakpoint_k or 1.5
        self._min_size = min_chunk_size or 100
        self._max_size = max_chunk_size or 1000

        logger.info(
            f"SemanticChunker initialized — "
            f"k={self._k}, min={self._min_size}, max={self._max_size}"
        )

    # ── Public API ───────────────────────────────────────────────────

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """
        Split a list of Documents into semantic chunks.

        This is the primary entry point. It processes documents
        grouped by source file to maintain cross-page continuity
        (a concept that starts on page 5 and continues on page 6
        should NOT be split at the page boundary).

        Args:
            documents: Cleaned Document objects from TextCleaner.

        Returns:
            List of Chunk objects with deterministic IDs and
            rich metadata. Ready for vector store ingestion.
        """
        if not documents:
            logger.warning("No documents to chunk.")
            return []

        # ── Group documents by source file ───────────────────────────
        # We concatenate all pages of the same PDF into one text block
        # BEFORE chunking, so the semantic chunker can detect topic
        # shifts across page boundaries (not just within pages).
        source_groups: dict[str, list[Document]] = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            source_groups.setdefault(source, []).append(doc)

        all_chunks: list[Chunk] = []

        for source_name, source_docs in source_groups.items():
            logger.info(
                f"Chunking [bold green]{source_name}[/bold green] "
                f"({len(source_docs)} pages)..."
            )

            # Sort by page number to maintain reading order
            source_docs.sort(key=lambda d: d.metadata.get("page", 0))

            chunks = self._chunk_single_document(source_name, source_docs)
            all_chunks.extend(chunks)

            logger.info(
                f"  -> {len(chunks)} semantic chunks created "
                f"from {source_name}"
            )

        # ── Pipeline summary ─────────────────────────────────────────
        if all_chunks:
            sizes = [len(c.text) for c in all_chunks]
            logger.info(
                f"[bold]Chunking complete[/bold]: "
                f"{len(all_chunks)} total chunks | "
                f"Size: min={min(sizes)}, avg={int(np.mean(sizes))}, "
                f"max={max(sizes)} chars"
            )

        return all_chunks

    # ── Core Algorithm ───────────────────────────────────────────────

    def _chunk_single_document(
        self,
        source_name: str,
        source_docs: list[Document],
    ) -> list[Chunk]:
        """
        Apply hybrid Header + Semantic chunking to all pages of a single PDF.

        Two-pass pipeline:
            Pass 1: MarkdownHeaderTextSplitter splits at H1/H2/H3 boundaries.
                    Each section gets metadata (section, subsection, etc.).
            Pass 2: Sections larger than max_chunk_size are further split
                    using the semantic chunker (embedding-based breakpoints).

        Post-processing:
            - Breadcrumb (heading path) is prepended to each chunk text.
            - Size guardrails enforce min/max sizes.
            - Deterministic IDs are assigned.

        Args:
            source_name: Filename of the source PDF.
            source_docs: All pages of this PDF, sorted by page number.

        Returns:
            List of Chunk objects for this document.
        """
        # -- Build page-aware text stream --
        full_text = ""
        page_boundaries: list[tuple[int, int, int]] = []

        for doc in source_docs:
            start = len(full_text)
            full_text += doc.page_content + "\n\n"
            end = len(full_text)
            page_boundaries.append((
                start, end, doc.metadata.get("page", 0)
            ))

        # ==============================================================
        # PASS 1: Header-aware split (MarkdownHeaderTextSplitter)
        # ==============================================================
        header_sections = _split_by_headers(full_text)
        logger.debug(
            f"  {source_name}: Pass 1 (headers) produced "
            f"{len(header_sections)} section(s)"
        )

        # ==============================================================
        # PASS 2: Semantic split for oversized sections
        # ==============================================================
        final_section_chunks: list[dict] = []  # {"text": str, "metadata": dict}

        for section in header_sections:
            section_text = section["text"]
            section_meta = section["metadata"]

            if len(section_text) <= self._max_size:
                # Section fits within size limit - keep as-is
                final_section_chunks.append(section)
            else:
                # Section too large - apply semantic sub-splitting
                logger.debug(
                    f"  Section '{section_meta.get('section', '?')}' "
                    f"is {len(section_text)} chars - applying semantic split"
                )
                sub_chunks = self._semantic_split_text(section_text)
                for sub_text in sub_chunks:
                    final_section_chunks.append({
                        "text": sub_text,
                        "metadata": section_meta,  # Inherit section metadata
                    })

        # ==============================================================
        # POST-PROCESSING: Breadcrumb injection + size guardrails
        # ==============================================================
        raw_texts = []
        section_metas = []
        for chunk_data in final_section_chunks:
            text = chunk_data["text"]
            meta = chunk_data["metadata"]

            # Prepend breadcrumb to chunk text for richer embeddings
            breadcrumb = _build_breadcrumb(meta)
            enriched_text = f"{breadcrumb}\n{text}" if breadcrumb else text

            raw_texts.append(enriched_text)
            section_metas.append(meta)

        # Apply size guardrails
        sized_chunks = self._apply_size_guardrails(raw_texts)

        # Build Chunk objects with metadata
        result: list[Chunk] = []
        for idx, chunk_text in enumerate(sized_chunks):
            # Determine which pages this chunk spans
            page_start, page_end = self._find_page_span(
                chunk_text, full_text, page_boundaries
            )

            # Get section metadata from the corresponding pre-guardrail chunk
            # (best-effort mapping since guardrails may merge/split)
            meta_idx = min(idx, len(section_metas) - 1) if section_metas else 0
            sec_meta = section_metas[meta_idx] if section_metas else {}

            chunk_id = self._generate_chunk_id(source_name, idx)

            result.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                metadata={
                    "source": source_name,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_index": idx,
                    "num_sentences": chunk_text.count(". ") + 1,
                    "char_count": len(chunk_text),
                    "section": sec_meta.get("section", ""),
                    "subsection": sec_meta.get("subsection", ""),
                    "subsubsection": sec_meta.get("subsubsection", ""),
                },
            ))

        return result

    # -- Semantic sub-splitting (for oversized header sections) ---------

    def _semantic_split_text(self, text: str) -> list[str]:
        """
        Split a single text block using embedding-based semantic chunking.

        This is the Pass 2 algorithm: tokenize into sentences, embed,
        find breakpoints via adaptive threshold, merge at breakpoints.

        Used only for header-sections that exceed max_chunk_size.
        """
        sentences = _split_into_sentences(text)

        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []

        # Embed and find breakpoints
        sentence_embeddings = self._engine.embed_texts(sentences)
        similarities = self._compute_consecutive_similarities(sentence_embeddings)
        breakpoints = self._find_breakpoints(similarities)

        # Merge at breakpoints
        raw_chunks = self._merge_sentences_at_breakpoints(sentences, breakpoints)

        return raw_chunks

    # ── Similarity Computation ───────────────────────────────────────

    @staticmethod
    def _compute_consecutive_similarities(
        embeddings: np.ndarray,
    ) -> np.ndarray:
       
        # Efficient vectorized dot product between consecutive pairs
        # This avoids a Python loop over potentially thousands of sentences.
        similarities = np.array([
            np.dot(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ])

        return similarities

    def _find_breakpoints(self, similarities: np.ndarray) -> list[int]:
        """
        Identify positions where topic shifts occur.

        A breakpoint at position i means that sentences i and i+1
        belong to DIFFERENT topics and should be in separate chunks.

        The threshold is adaptive:
            T = μ(similarities) - k · σ(similarities)

        Positions where similarity < T are breakpoints.

        Args:
            similarities: Array of consecutive-pair cosine similarities.

        Returns:
            Sorted list of breakpoint indices.
        """
        if len(similarities) == 0:
            return []

        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        # The adaptive threshold: sentences with similarity more than
        # k standard deviations below the mean are at topic boundaries.
        threshold = mean_sim - (self._k * std_sim)

        logger.debug(
            f"  Similarity stats: μ={mean_sim:.4f}, σ={std_sim:.4f}, "
            f"threshold={threshold:.4f} (k={self._k})"
        )

        # Find all positions below the threshold
        breakpoints = list(np.where(similarities < threshold)[0])

        return breakpoints

    # ── Sentence Merging ─────────────────────────────────────────────

    @staticmethod
    def _merge_sentences_at_breakpoints(
        sentences: list[str],
        breakpoints: list[int],
    ) -> list[str]:
        """
        Group sentences into chunks based on breakpoint positions.

        Sentences between consecutive breakpoints are joined with a
        single space to form coherent text blocks.

        Example:
            sentences = [s0, s1, s2, s3, s4, s5]
            breakpoints = [1, 4]
            Result: ["s0 s1", "s2 s3 s4", "s5"]
            (breaks AFTER index 1 and AFTER index 4)

        Args:
            sentences:   All sentences in reading order.
            breakpoints: Indices where breaks occur (break AFTER this index).

        Returns:
            List of merged chunk texts.
        """
        if not breakpoints:
            # No breakpoints -> entire document is one chunk
            return [" ".join(sentences)]

        chunks: list[str] = []
        start_idx = 0

        for bp in sorted(breakpoints):
            # Merge sentences from start_idx to bp (inclusive)
            chunk_sentences = sentences[start_idx : bp + 1]
            if chunk_sentences:
                chunks.append(" ".join(chunk_sentences))
            start_idx = bp + 1

        # Don't forget the last chunk (after the final breakpoint)
        if start_idx < len(sentences):
            chunks.append(" ".join(sentences[start_idx:]))

        return chunks

    # ── Size Guardrails ──────────────────────────────────────────────

    def _apply_size_guardrails(
        self, chunks: list[str]
    ) -> list[str]:
        """
        Enforce minimum and maximum chunk sizes.

        - Chunks below min_chunk_size are merged with the NEXT chunk
          (or the previous chunk if it's the last one).
        - Chunks above max_chunk_size are split at the nearest
          sentence boundary.

        This prevents two degenerate cases:
            1. Tiny chunks (1-2 sentences) that don't carry enough
               context for meaningful retrieval.
            2. Huge chunks that exceed the embedding model's effective
               context window (~256 tokens for MiniLM).

        Args:
            chunks: Raw merged chunk texts from breakpoint splitting.

        Returns:
            Size-adjusted chunk texts.
        """
        if not chunks:
            return chunks

        # ── Phase A: Merge undersized chunks ─────────────────────────
        merged: list[str] = []
        buffer = ""

        for chunk in chunks:
            if buffer:
                # We have leftover text from a previous undersized chunk
                combined = buffer + " " + chunk
                if len(combined) <= self._max_size:
                    buffer = combined
                else:
                    # Buffer + current exceeds max -> flush buffer, start new
                    merged.append(buffer)
                    buffer = chunk
            else:
                buffer = chunk

            # Flush buffer if it's large enough
            if len(buffer) >= self._min_size:
                merged.append(buffer)
                buffer = ""

        # Handle any remaining buffer
        if buffer:
            if merged and len(merged[-1]) + len(buffer) + 1 <= self._max_size:
                # Merge with the last chunk if it fits
                merged[-1] = merged[-1] + " " + buffer
            else:
                merged.append(buffer)

        # ── Phase B: Split oversized chunks ──────────────────────────
        final: list[str] = []

        for chunk in merged:
            if len(chunk) <= self._max_size:
                final.append(chunk)
            else:
                # Split at sentence boundaries within the oversized chunk
                sub_sentences = _split_into_sentences(chunk)
                sub_buffer = ""

                for sent in sub_sentences:
                    candidate = (sub_buffer + " " + sent).strip()
                    if len(candidate) <= self._max_size:
                        sub_buffer = candidate
                    else:
                        if sub_buffer:
                            final.append(sub_buffer)
                        sub_buffer = sent

                if sub_buffer:
                    final.append(sub_buffer)

        return final

    # ── Metadata Helpers ─────────────────────────────────────────────

    @staticmethod
    def _find_page_span(
        chunk_text: str,
        full_text: str,
        page_boundaries: list[tuple[int, int, int]],
    ) -> tuple[int, int]:
        """
        Determine which pages a chunk spans by finding its position
        in the original concatenated text.

        Args:
            chunk_text:       The chunk's text content.
            full_text:        The full concatenated document text.
            page_boundaries:  List of (start_char, end_char, page_num).

        Returns:
            Tuple of (page_start, page_end). Both are zero-indexed
            page numbers from the original PDF.
        """
        # Find where this chunk appears in the full text
        chunk_start = full_text.find(chunk_text[:100])
        if chunk_start == -1:
            # Fallback: can't locate chunk (shouldn't happen)
            return (0, 0)

        chunk_end = chunk_start + len(chunk_text)

        page_start = 0
        page_end = 0

        for start, end, page_num in page_boundaries:
            if start <= chunk_start < end:
                page_start = page_num
            if start < chunk_end <= end:
                page_end = page_num

        return (page_start, page_end)

    @staticmethod
    def _generate_chunk_id(source_name: str, chunk_index: int) -> str:
        """
        Generate a deterministic chunk ID for idempotent storage.

        The ID is a SHA-256 hash of the source filename and chunk index.
        This means re-running the pipeline on the same PDF produces
        the same IDs, enabling upsert (update-or-insert) in ChromaDB
        without creating duplicates.

        Args:
            source_name: PDF filename (e.g., 'os_notes.pdf').
            chunk_index: Sequential index of this chunk in the document.

        Returns:
            64-character hex string (SHA-256 hash).
        """
        key = f"{source_name}::chunk_{chunk_index}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _create_single_chunk(
        self,
        text: str,
        source_name: str,
        source_docs: list[Document],
    ) -> list[Chunk]:
        """
        Handle the edge case where a document has ≤1 sentence.

        Returns the entire document as a single chunk.
        """
        if not text.strip():
            return []

        pages = [d.metadata.get("page", 0) for d in source_docs]

        return [Chunk(
            text=text.strip(),
            chunk_id=self._generate_chunk_id(source_name, 0),
            metadata={
                "source": source_name,
                "page_start": min(pages) if pages else 0,
                "page_end": max(pages) if pages else 0,
                "chunk_index": 0,
                "num_sentences": text.count(". ") + 1,
                "char_count": len(text),
            },
        )]
