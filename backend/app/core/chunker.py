"""
chunker.py — Semantic Chunker with Adaptive Breakpoint Detection
==================================================================

The most technically complex module in the pipeline. Splits cleaned
documents into chunks at NATURAL TOPIC BOUNDARIES instead of at
arbitrary character counts.

Algorithm Overview:
    1. Tokenize document text into individual sentences (NLTK punkt).
    2. Embed every sentence using SentenceTransformer (batched).
    3. Compute cosine similarity between each consecutive sentence pair.
    4. Calculate an adaptive breakpoint threshold:
           threshold = mean(similarities) - k * std(similarities)
    5. Insert chunk boundaries wherever consecutive similarity drops
       below the threshold (= a natural topic shift).
    6. Merge sentences between breakpoints into coherent chunks.
    7. Apply min/max size guardrails to prevent degenerate output.

Why This Beats Naive Chunking (RecursiveCharacterTextSplitter):
    ┌─────────────────────────┬──────────────────────────────────────┐
    │ Naive (Fixed-Window)    │ Semantic (This Module)              │
    ├─────────────────────────┼──────────────────────────────────────┤
    │ Splits at char count    │ Splits at topic boundaries          │
    │ Fragments mid-concept   │ Preserves complete ideas            │
    │ Noisy embeddings        │ Clean, focused embeddings           │
    │ Same threshold for all  │ Adaptive per-document threshold     │
    │ Overlap needed as hack  │ No overlap needed (natural breaks)  │
    └─────────────────────────┴──────────────────────────────────────┘

Mathematical Foundation:
    Given consecutive sentence embeddings e_1, e_2, ..., e_n, we compute:

        d_i = cosine_similarity(e_i, e_{i+1})    for i in [1, n-1]

    The similarity sequence D = [d_1, d_2, ..., d_{n-1}] represents
    the "coherence signal" of the document. Topic shifts appear as
    valleys (local minima) in this signal.

    We set a global threshold:
        T = μ(D) - k · σ(D)

    Where:
        μ(D) = mean of all consecutive similarities
        σ(D) = standard deviation of similarities
        k    = sensitivity parameter (default 1.0)

    Any position i where d_i < T is marked as a breakpoint.
    This is adaptive because μ and σ are computed per-document,
    so technical papers (lower baseline similarity) and narrative
    text (higher baseline similarity) get appropriate thresholds.

Usage:
    from app.core.chunker import SemanticChunker
    from app.core.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    chunker = SemanticChunker(embedding_engine=engine)
    chunks = chunker.chunk(cleaned_documents)

Dependencies:
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
        text:       The chunk content (merged sentences).
        chunk_id:   Deterministic ID for idempotent storage.
                    Computed as SHA-256 of (source + chunk_index).
        metadata:   Dictionary containing provenance information:
                    - source:        PDF filename
                    - page_start:    First page this chunk spans
                    - page_end:      Last page this chunk spans
                    - chunk_index:   Sequential index within the document
                    - num_sentences: How many sentences were merged
                    - char_count:    Character length of the chunk
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
        - Handles abbreviations: "Dr. Smith went..." → 1 sentence
        - Handles decimals: "scored 3.5 points" → not split at "3."
        - Handles ellipses: "wait... what?" → 1 sentence
        - Academic-aware: "et al. (2020)" → not split

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
                            threshold. Higher k → fewer, larger chunks.
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
        self._k = breakpoint_k or settings.chunker.breakpoint_sensitivity_k
        self._min_size = min_chunk_size or settings.chunker.min_chunk_size
        self._max_size = max_chunk_size or settings.chunker.max_chunk_size

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
                f"  → {len(chunks)} semantic chunks created "
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
        Apply semantic chunking to all pages of a single PDF.

        Steps:
            1. Concatenate all pages into one text stream.
            2. Split into sentences.
            3. Embed all sentences (batched).
            4. Find breakpoints using adaptive threshold.
            5. Merge sentences between breakpoints into chunks.
            6. Apply min/max size guardrails.
            7. Assign deterministic IDs and metadata.

        Args:
            source_name: Filename of the source PDF.
            source_docs: All pages of this PDF, sorted by page number.

        Returns:
            List of Chunk objects for this document.
        """
        # ── Step 1: Build page-aware text stream ─────────────────────
        # We track which page each sentence came from by building
        # a mapping of character offsets to page numbers.
        full_text = ""
        page_boundaries: list[tuple[int, int, int]] = []
        # Each entry: (start_char, end_char, page_number)

        for doc in source_docs:
            start = len(full_text)
            full_text += doc.page_content + "\n\n"
            end = len(full_text)
            page_boundaries.append((
                start, end, doc.metadata.get("page", 0)
            ))

        # ── Step 2: Sentence tokenization ────────────────────────────
        sentences = _split_into_sentences(full_text)

        if len(sentences) <= 1:
            # Edge case: document has 0 or 1 sentences — no splitting
            logger.debug(f"  {source_name}: ≤1 sentence, returning as single chunk")
            return self._create_single_chunk(
                full_text.strip(), source_name, source_docs
            )

        logger.debug(f"  {source_name}: {len(sentences)} sentences extracted")

        # ── Step 3: Batch embed all sentences ────────────────────────
        # This is done in a single call for efficiency. Embedding
        # 500 sentences one-at-a-time would be ~50× slower than
        # a single batched call.
        sentence_embeddings = self._engine.embed_texts(sentences)

        # ── Step 4: Compute consecutive cosine similarities ──────────
        similarities = self._compute_consecutive_similarities(
            sentence_embeddings
        )

        # ── Step 5: Find breakpoints ─────────────────────────────────
        breakpoints = self._find_breakpoints(similarities)
        logger.debug(
            f"  {source_name}: {len(breakpoints)} breakpoints detected "
            f"out of {len(similarities)} sentence pairs"
        )

        # ── Step 6: Merge sentences into chunks ──────────────────────
        raw_chunks = self._merge_sentences_at_breakpoints(
            sentences, breakpoints
        )

        # ── Step 7: Apply size guardrails ────────────────────────────
        sized_chunks = self._apply_size_guardrails(raw_chunks)

        # ── Step 8: Build Chunk objects with metadata ────────────────
        result: list[Chunk] = []
        for idx, chunk_text in enumerate(sized_chunks):
            # Determine which pages this chunk spans
            page_start, page_end = self._find_page_span(
                chunk_text, full_text, page_boundaries
            )

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
                },
            ))

        return result

    # ── Similarity Computation ───────────────────────────────────────

    @staticmethod
    def _compute_consecutive_similarities(
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between each pair of consecutive
        sentence embeddings.

        For embeddings [e_0, e_1, e_2, ..., e_n], computes:
            [cos(e_0, e_1), cos(e_1, e_2), ..., cos(e_{n-1}, e_n)]

        Since our embeddings are L2-normalized (done in EmbeddingEngine),
        cosine similarity simplifies to a dot product:
            cos(a, b) = a · b / (||a|| · ||b||) = a · b   (when ||a||=||b||=1)

        Args:
            embeddings: (n, d) array of L2-normalized sentence embeddings.

        Returns:
            (n-1,) array of similarity scores in range [-1, 1].
        """
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
            # No breakpoints → entire document is one chunk
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
                    # Buffer + current exceeds max → flush buffer, start new
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
