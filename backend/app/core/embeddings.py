"""
embeddings.py — Local Embedding Engine (SentenceTransformer)
=============================================================

Thin, focused wrapper around SentenceTransformer that provides:
    1. Single-load model caching (load once, embed forever).
    2. Batch encoding with L2 normalization for cosine similarity.
    3. ChromaDB-compatible EmbeddingFunction interface.

Model: all-MiniLM-L6-v2
    - 384 dimensions, 22M parameters, ~80MB on disk.
    - Trained on 1B+ sentence pairs (NLI + STS benchmarks).
    - MTEB score ~63 — best speed/quality ratio for academic text.
    - Runs comfortably on CPU in <100ms per query.

Why we normalize embeddings:
    Cosine similarity between vectors a and b is:
        cos(a, b) = (a · b) / (||a|| · ||b||)

    When both vectors are L2-normalized (||a|| = ||b|| = 1), this
    simplifies to a dot product: cos(a, b) = a · b

    This is critical because:
    - ChromaDB uses cosine distance by default.
    - Our semantic chunker computes dot products for speed.
    - Normalized embeddings make similarity scores directly
      interpretable as values in [-1, 1].

Usage:
    from app.core.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    vectors = engine.embed_texts(["hello world", "foo bar"])
    query_vec = engine.embed_query("what is paging?")
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingEngine:
    """
    Local embedding engine backed by SentenceTransformer.

    The model is loaded lazily on the first embed call and cached
    for the lifetime of the process. This avoids the 2-3 second
    model load penalty if embeddings are never used (e.g., in
    unit tests that mock this class).

    Thread Safety:
        SentenceTransformer.encode() is thread-safe for inference.
        Multiple threads can call embed_texts() concurrently.

    Args:
        model_name: HuggingFace model identifier. Defaults to config.
        device:     'cpu' or 'cuda'. Defaults to config.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name or settings.embedding.model_name
        self._device = device or settings.embedding.device
        self._model = None  # Lazy-loaded

        logger.info(
            f"EmbeddingEngine configured — "
            f"model='{self._model_name}', device='{self._device}'"
        )

    # ── Lazy Model Loading ───────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Load the SentenceTransformer model into memory.

        Called automatically on the first embed call. The model is
        downloaded from HuggingFace on first use (~80MB for MiniLM)
        and cached locally at ~/.cache/torch/sentence_transformers/.

        After the first download, this works fully offline.
        """
        if self._model is not None:
            return

        logger.info(
            f"Loading embedding model '{self._model_name}' "
            f"on {self._device}..."
        )

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self._model_name,
            device=self._device,
        )

        # Validate output dimension matches config
        test_embedding = self._model.encode(["test"], normalize_embeddings=True)
        actual_dim = test_embedding.shape[1]
        expected_dim = settings.embedding.embedding_dimension

        if actual_dim != expected_dim:
            logger.warning(
                f"Embedding dimension mismatch! "
                f"Model outputs {actual_dim}d, config expects {expected_dim}d. "
                f"Update EmbeddingConfig.embedding_dimension."
            )

        logger.info(
            f"Model loaded — {actual_dim}d embeddings, "
            f"device={self._device}"
        )

    # ── Public API ───────────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of texts into L2-normalized embeddings.

        This is the workhorse method used by both the semantic chunker
        (embedding hundreds of sentences) and the vector store
        (embedding chunks for storage).

        Batch encoding is critical for performance:
            - 500 sentences batched: ~2 seconds
            - 500 sentences one-at-a-time: ~50 seconds

        Args:
            texts: List of text strings to encode.

        Returns:
            numpy array of shape (len(texts), embedding_dimension).
            Each row is an L2-normalized embedding vector.
        """
        self._load_model()

        if not texts:
            return np.array([])

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=settings.embedding.normalize,
            show_progress_bar=len(texts) > 100,  # Only for large batches
            batch_size=64,  # Optimal for CPU inference
        )

        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.

        Convenience method for retrieval. Returns a 1D vector
        (not a 2D array) for direct use in similarity computation.

        Args:
            query: The search query text.

        Returns:
            1D numpy array of shape (embedding_dimension,).
        """
        self._load_model()

        embedding = self._model.encode(
            [query],
            normalize_embeddings=settings.embedding.normalize,
        )

        return np.array(embedding[0])

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        return settings.embedding.embedding_dimension

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name


# ═══════════════════════════════════════════════════════════════════════
# ChromaDB-Compatible Embedding Function
# ═══════════════════════════════════════════════════════════════════════

class ChromaEmbeddingFunction:
    """
    Adapter that wraps EmbeddingEngine to conform to ChromaDB's
    EmbeddingFunction protocol.

    ChromaDB requires embedding functions to implement __call__()
    returning a list of lists. This adapter bridges our numpy-based
    EmbeddingEngine to ChromaDB's expected interface.

    Usage:
        chroma_fn = ChromaEmbeddingFunction(engine)
        collection = client.get_or_create_collection(
            embedding_function=chroma_fn
        )
    """

    def __init__(self, engine: EmbeddingEngine) -> None:
        self._engine = engine

    def __call__(self, input: list[str]) -> list[list[float]]:
        """
        Encode texts in the format ChromaDB expects.

        Args:
            input: List of text strings (ChromaDB's parameter name).

        Returns:
            List of embedding vectors as Python lists of floats.
        """
        embeddings = self._engine.embed_texts(input)
        return embeddings.tolist()
