"""
vector_store.py — Persistent ChromaDB Vector Store
====================================================

Manages the lifecycle of the ChromaDB collection: creating it,
upserting chunks with embeddings and metadata, querying, and
providing collection statistics.

Key Design Decisions:
    1. Persistent storage — ChromaDB data survives process restarts.
       Data is written to backend/data/chroma_db/.
    2. Idempotent ingestion — Chunk IDs are deterministic (SHA-256
       of source + index). Re-running the pipeline on the same PDFs
       performs upserts, not inserts. No duplicates ever.
    3. Embedding delegation — We pass our EmbeddingEngine to ChromaDB
       via the ChromaEmbeddingFunction adapter so ChromaDB can embed
       query strings automatically during search.

Usage:
    from app.core.vector_store import ChromaVectorStore
    from app.core.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    store = ChromaVectorStore(engine)
    store.add_chunks(chunks)
    results = store.query("what is paging?", n_results=5)
"""

from __future__ import annotations

from typing import Any, Optional

from app.config import settings
from app.core.chunker import Chunk
from app.core.embeddings import ChromaEmbeddingFunction, EmbeddingEngine
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStore:
    """
    Persistent vector store backed by ChromaDB.

    Handles the full storage lifecycle: collection creation, chunk
    upsertion with metadata, similarity queries, and cleanup.

    The collection is created with cosine distance metric to match
    our L2-normalized embeddings (cosine distance = 1 - cosine similarity).

    Args:
        embedding_engine: Initialized EmbeddingEngine for encoding.
        persist_dir:      Directory for ChromaDB's persistent storage.
        collection_name:  Name of the ChromaDB collection.
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._engine = embedding_engine
        self._persist_dir = persist_dir or str(settings.CHROMA_PATH)
        self._collection_name = collection_name or "rag_study_chunks"

        # ── Initialize ChromaDB client ───────────────────────────────
        import chromadb

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
        )

        # ── Get or create collection ─────────────────────────────────
        # ChromaDB's get_or_create is idempotent — safe to call on
        # every pipeline run without checking if collection exists.
        self._embedding_fn = ChromaEmbeddingFunction(self._engine)

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_fn,
            metadata={
                "hnsw:space": "cosine",
            },
        )

        count = self._collection.count()
        logger.info(
            f"ChromaVectorStore initialized — "
            f"collection='{self._collection_name}', "
            f"existing chunks={count}, "
            f"persist_dir='{self._persist_dir}'"
        )

    # ── Public API ───────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> int:
        """
        Upsert chunks into the ChromaDB collection.

        Uses upsert (not add) to handle idempotent re-ingestion.
        If a chunk with the same ID already exists, it is overwritten
        with the new content and metadata.

        Processing is done in batches to avoid memory issues with
        very large document sets.

        Args:
            chunks:     List of Chunk objects from the SemanticChunker.
            batch_size: Number of chunks to upsert per ChromaDB call.
                        Larger batches are faster but use more memory.

        Returns:
            Number of chunks successfully upserted.
        """
        if not chunks:
            logger.warning("No chunks to store.")
            return 0

        logger.info(f"Upserting {len(chunks)} chunks into ChromaDB...")

        total_upserted = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [c.metadata for c in batch]

            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

            total_upserted += len(batch)
            logger.debug(
                f"  Batch {i // batch_size + 1}: "
                f"upserted {len(batch)} chunks "
                f"({total_upserted}/{len(chunks)})"
            )

        final_count = self._collection.count()
        logger.info(
            f"[bold]Upsert complete[/bold]: "
            f"{total_upserted} chunks processed, "
            f"collection now has {final_count} total chunks"
        )

        return total_upserted

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where_filter: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Query the collection for similar chunks.

        This is the low-level query method. For advanced retrieval
        with MMR and thresholding, use the AdvancedRetriever class.

        Args:
            query_text:   The search query string.
            n_results:    Maximum number of results to return.
            where_filter: Optional metadata filter (ChromaDB where clause).
                          Example: {"source": "os_notes.pdf"}

        Returns:
            ChromaDB query result dict with keys:
            - 'ids': List of chunk IDs.
            - 'documents': List of chunk texts.
            - 'metadatas': List of metadata dicts.
            - 'distances': List of cosine distances (lower = more similar).
        """
        query_params: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(n_results, self._collection.count()),
        }

        if where_filter:
            query_params["where"] = where_filter

        results = self._collection.query(**query_params)

        return results

    def query_by_embedding(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where_filter: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Query the collection using a pre-computed embedding vector.

        Used by the AdvancedRetriever for MMR, where we need the
        raw query embedding for additional similarity computations.

        Args:
            query_embedding: Pre-computed embedding vector.
            n_results:       Maximum number of results.
            where_filter:    Optional metadata filter.

        Returns:
            ChromaDB query result dict.
        """
        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._collection.count()),
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }

        if where_filter:
            query_params["where"] = where_filter

        return self._collection.query(**query_params)

    def get_collection_stats(self) -> dict[str, Any]:
        """
        Return summary statistics about the collection.

        Useful for pipeline_runner output and debugging.

        Returns:
            Dict with 'count', 'collection_name', 'persist_dir'.
        """
        count = self._collection.count()

        stats = {
            "collection_name": self._collection_name,
            "total_chunks": count,
            "persist_dir": self._persist_dir,
            "distance_metric": "cosine",
            "embedding_model": self._engine.model_name,
            "embedding_dimension": self._engine.dimension,
        }

        # Sample metadata from first few chunks for summary
        if count > 0:
            sample = self._collection.peek(limit=min(5, count))
            sources = set()
            for meta in sample.get("metadatas", []):
                if meta and "source" in meta:
                    sources.add(meta["source"])
            stats["sample_sources"] = list(sources)

        return stats

    def delete_collection(self) -> None:
        """
        Delete the entire collection. Irreversible.

        Use this to force a full re-ingestion from scratch.
        """
        logger.warning(
            f"Deleting collection '{self._collection_name}'..."
        )
        self._client.delete_collection(self._collection_name)

        # Re-create empty collection for immediate reuse
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_fn,
            metadata={
                "hnsw:space": "cosine",
            },
        )

        logger.info("Collection deleted and re-created (empty).")

    @property
    def count(self) -> int:
        """Return the current number of chunks in the collection."""
        return self._collection.count()
