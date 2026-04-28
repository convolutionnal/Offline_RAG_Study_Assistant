"""
config.py — Centralized Configuration for the RAG Pipeline
===========================================================

All magic numbers, file paths, model names, and tunable parameters
live here. Every other module imports from this single source of truth.

Design Decision:
    We use a frozen dataclass instead of environment variables because
    this is an offline system with no deployment variance. A dataclass
    gives us type safety, IDE autocompletion, and immutability guarantees
    that a .env file cannot provide.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ── Resolve project root paths ──────────────────────────────────────────
# backend/ is the root of the Python project.
# All other paths are computed relative to this anchor.
_BACKEND_DIR = Path(__file__).resolve().parent.parent  # .../backend/
_PROJECT_ROOT = _BACKEND_DIR.parent                     # .../Offline_RAG_Study_Assistant/


@dataclass(frozen=True)
class PathConfig:
    """
    All filesystem paths used by the pipeline.

    Frozen dataclass ensures paths cannot be accidentally mutated
    at runtime. Every path is an absolute Path object.
    """

    # ── Input ────────────────────────────────────────────────────────
    # Directory containing source PDF files for ingestion.
    pdf_input_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "sample_docs"
    )

    # ── Data Storage ─────────────────────────────────────────────────
    # Root directory for all pipeline-generated data.
    data_dir: Path = field(
        default_factory=lambda: _BACKEND_DIR / "data"
    )

    # Persistent ChromaDB storage directory.
    chroma_db_dir: Path = field(
        default_factory=lambda: _BACKEND_DIR / "data" / "chroma_db"
    )

    # Directory for extracted raw text (intermediate, for debugging).
    extracted_dir: Path = field(
        default_factory=lambda: _BACKEND_DIR / "data" / "extracted"
    )

    # Directory for processed/cleaned text (intermediate, for debugging).
    processed_dir: Path = field(
        default_factory=lambda: _BACKEND_DIR / "data" / "processed"
    )

    # Directory for uploaded files (used by the FastAPI layer in Phase 2).
    uploads_dir: Path = field(
        default_factory=lambda: _BACKEND_DIR / "data" / "uploads"
    )

    def ensure_directories(self) -> None:
        """
        Create all data directories if they don't exist.

        Called once at pipeline startup to avoid FileNotFoundError
        during ingestion. Uses exist_ok=True for idempotency.
        """
        for dir_path in [
            self.data_dir,
            self.chroma_db_dir,
            self.extracted_dir,
            self.processed_dir,
            self.uploads_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Configuration for the SentenceTransformer embedding model.

    We default to 'all-MiniLM-L6-v2' — a 384-dimensional model with
    22M parameters that runs comfortably on CPU. It offers the best
    speed/quality trade-off for academic text retrieval.

    MTEB Benchmark Score: ~63 (competitive with models 4× its size).
    """

    # HuggingFace model identifier. Downloaded once, cached locally.
    model_name: str = "all-MiniLM-L6-v2"

    # Output dimensionality of the model. Must match the model's
    # actual output dimension — used for validation only.
    embedding_dimension: int = 384

    # Device for inference. 'cpu' is the safe default for offline use.
    # Set to 'cuda' if a compatible GPU is available.
    device: str = "cpu"

    # Whether to L2-normalize embeddings. Must be True when using
    # cosine similarity (which ChromaDB uses by default).
    normalize: bool = True


@dataclass(frozen=True)
class ChunkerConfig:
    """
    Configuration for the Semantic Chunker.

    The chunker splits documents at natural topic boundaries by
    comparing embedding similarity between consecutive sentences.

    Breakpoint Threshold Formula:
        threshold = mean(similarities) - k * std(similarities)

    Where 'k' is the breakpoint_percentile_sensitivity below.
    """

    # ── Breakpoint Sensitivity ───────────────────────────────────────
    # Higher k → fewer breaks → larger chunks (more lenient).
    # Lower k  → more breaks  → smaller chunks (more aggressive).
    # k=1.0 means: break when similarity is 1 std dev below the mean.
    breakpoint_sensitivity_k: float = 1.0

    # ── Chunk Size Guardrails ────────────────────────────────────────
    # Minimum characters per chunk. Chunks below this are merged
    # with their nearest neighbor to prevent degenerate fragments.
    min_chunk_size: int = 100

    # Maximum characters per chunk. Chunks above this are split
    # at the nearest sentence boundary to prevent context overflow.
    max_chunk_size: int = 2000


@dataclass(frozen=True)
class RetrieverConfig:
    """
    Configuration for the Advanced Retriever.

    Supports three retrieval modes:
        1. top_k     — Return K most similar chunks.
        2. threshold — Return all chunks above a similarity score.
        3. mmr       — Maximum Marginal Relevance (diversity-aware).
    """

    # Default number of results to return.
    top_k: int = 5

    # Number of candidates to fetch before MMR re-ranking.
    # Should be > top_k to give MMR enough diversity headroom.
    mmr_candidate_count: int = 20

    # ── MMR Trade-off ────────────────────────────────────────────────
    # λ (lambda) in the MMR formula:
    #   MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected))
    #
    # λ=1.0 → pure relevance (identical to top_k).
    # λ=0.0 → pure diversity (ignores relevance entirely).
    # λ=0.7 → strong relevance bias with meaningful diversity.
    mmr_lambda: float = 0.7

    # ── Similarity Threshold ─────────────────────────────────────────
    # Minimum cosine similarity score to include a result.
    # Chunks scoring below this are considered irrelevant noise.
    # Range: [0.0, 1.0] for normalized embeddings.
    similarity_threshold: float = 0.3


@dataclass(frozen=True)
class VectorStoreConfig:
    """
    Configuration for the ChromaDB vector store.
    """

    # Name of the ChromaDB collection. Changing this creates a new
    # isolated collection (useful for A/B testing different chunkers).
    collection_name: str = "rag_study_chunks"

    # Distance metric used by ChromaDB for similarity search.
    # 'cosine' is the standard choice for normalized embeddings.
    distance_metric: str = "cosine"


@dataclass(frozen=True)
class PipelineConfig:
    """
    Top-level configuration that aggregates all sub-configs.

    Usage:
        from app.config import PipelineConfig
        config = PipelineConfig()
        print(config.paths.pdf_input_dir)
        print(config.embedding.model_name)
    """

    paths: PathConfig = field(default_factory=PathConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)


# ── Module-level singleton ───────────────────────────────────────────────
# Import this directly: `from app.config import settings`
# Every module in the pipeline references the same instance.
settings = PipelineConfig()
