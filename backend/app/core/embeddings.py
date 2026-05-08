from __future__ import annotations

import numpy as np
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
# Core Embedding Engine
# ═══════════════════════════════════════════════════════════

class EmbeddingEngine:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name or settings.EMBEDDING_MODEL

        requested_device = device or "cpu"

        if requested_device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = "cuda"
                    logger.info(
                        f"CUDA available — GPU: {torch.cuda.get_device_name(0)}"
                    )
                else:
                    self._device = "cpu"
                    logger.warning(
                        "CUDA requested but not available - falling back to CPU"
                    )
            except ImportError:
                self._device = "cpu"
                logger.warning("PyTorch not found - using CPU")
        else:
            self._device = requested_device

        self._model = None

        logger.info(
            f"EmbeddingEngine configured - "
            f"model='{self._model_name}', device='{self._device}'"
        )

    # ─────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return

        logger.info(
            f"Loading embedding model '{self._model_name}' on {self._device}..."
        )

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self._model_name,
            device=self._device,
        )

        test_embedding = self._model.encode(["test"], normalize_embeddings=True)
        actual_dim = test_embedding.shape[1]
        expected_dim = 768 # Default for all-mpnet-base-v2

        if actual_dim != expected_dim:
            logger.warning(
                f"Embedding dimension mismatch! "
                f"Model outputs {actual_dim}d, config expects {expected_dim}d."
            )

        logger.info(
            f"Model loaded — {actual_dim}d embeddings, device={self._device}"
        )

    # ─────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self._load_model()

        if not texts:
            return np.array([])

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            batch_size=32,
        )

        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        self._load_model()

        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
        )

        return np.array(embedding[0])

    @property
    def dimension(self) -> int:
        return 768 # Default for all-mpnet-base-v2

    @property
    def model_name(self) -> str:
        return self._model_name

_engine_instance = None

def get_embedding_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EmbeddingEngine()
    return _engine_instance

# ═══════════════════════════════════════════════════════════
# ChromaDB Adapter
# ═══════════════════════════════════════════════════════════

class ChromaEmbeddingFunction:
    def __init__(self, engine: EmbeddingEngine):
        self.engine = engine

    def __call__(self, input):
        """
        Used by Chroma for embedding documents.
        Must return List[List[float]]
        """
        if isinstance(input, str):
            input = [input]

        embeddings = self.engine.embed_texts(input)
        return embeddings.tolist()

    def embed_query(self, input):
        """
        Used by Chroma for embedding queries.
        Must return List[List[float]]
        """
        # Fix: Chroma may send list instead of string
        if isinstance(input, list):
            input = input[0] if input else ""

        embedding = self.engine.embed_query(input)

        #  CRITICAL FIX: must be [[...]]
        return [embedding.tolist()]

    def name(self) -> str:
        return "custom-embedding-engine"