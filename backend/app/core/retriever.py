"""
retriever.py — Advanced Retrieval with MMR & Threshold Filtering
==================================================================

Goes beyond basic Top-K retrieval to provide three search modes:

1. Top-K Similarity:
   Return the K chunks with highest cosine similarity to the query.
   Simple, fast, but prone to redundancy.

2. Threshold-Filtered:
   Return ALL chunks above a minimum similarity score, regardless
   of count. Useful when you don't know how many relevant chunks exist.

3. Maximum Marginal Relevance (MMR):
   The advanced mode. Balances relevance and diversity to ensure the
   retrieved context covers different aspects of the query.

MMR Formula:
    MMR(q, D, S) = argmax_{d_i ∈ D\\S} [
        λ · sim(q, d_i)  -  (1-λ) · max_{d_j ∈ S} sim(d_i, d_j)
    ]

    Where:
        q = query embedding
        D = candidate set (initial top-K retrieval)
        S = already selected results
        λ = trade-off parameter (0 = pure diversity, 1 = pure relevance)

    Intuition: Each next result is chosen to be BOTH relevant to the
    query AND different from what we've already selected. This prevents
    the common failure mode where top-5 results all come from the same
    paragraph with slightly different overlapping windows.

Usage:
    from app.core.retriever import AdvancedRetriever

    retriever = AdvancedRetriever(vector_store, embedding_engine)
    results = retriever.retrieve("what is virtual memory?", mode="mmr")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from app.config import settings
from app.core.embeddings import EmbeddingEngine
from app.core.vector_store import ChromaVectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════

class RetrievalMode(str, Enum):
    """Supported retrieval strategies."""
    TOP_K = "top_k"
    THRESHOLD = "threshold"
    MMR = "mmr"


@dataclass
class RetrievalResult:
    """
    A single retrieval result with score and provenance metadata.

    Attributes:
        text:     The chunk text content.
        score:    Similarity score (higher = more relevant).
                  For cosine similarity: range [-1, 1].
        rank:     Position in the result list (1-indexed).
        metadata: Source document, page range, chunk index, etc.
    """
    text: str
    score: float
    rank: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:50].replace("\n", "\\n")
        return (
            f"Result(rank={self.rank}, score={self.score:.4f}, "
            f"source='{self.metadata.get('source', '?')}', "
            f"preview='{preview}...')"
        )


# ═══════════════════════════════════════════════════════════════════════
# Advanced Retriever
# ═══════════════════════════════════════════════════════════════════════

class AdvancedRetriever:
    """
    Multi-strategy retrieval engine with MMR support.

    Wraps the ChromaVectorStore with higher-level retrieval logic
    including re-ranking, diversity optimization, and score filtering.

    Args:
        vector_store:     Initialized ChromaVectorStore instance.
        embedding_engine: Initialized EmbeddingEngine for query encoding.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_engine: EmbeddingEngine,
    ) -> None:
        self._store = vector_store
        self._engine = embedding_engine

        logger.info("AdvancedRetriever initialized")

    # ── Public API ───────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        mode: str = "mmr",
        top_k: int | None = None,
        threshold: float | None = None,
        mmr_lambda: float | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks for a query using the specified strategy.

        This is the primary entry point for all retrieval operations.

        Args:
            query:         The search query string.
            mode:          Retrieval strategy: 'top_k', 'threshold', or 'mmr'.
            top_k:         Number of results to return. Defaults to config.
            threshold:     Minimum similarity score. Defaults to config.
            mmr_lambda:    MMR diversity parameter. Defaults to config.
            source_filter: Optional: restrict results to a specific PDF.

        Returns:
            List of RetrievalResult objects, sorted by relevance.

        Raises:
            ValueError: If an unknown retrieval mode is specified.
        """
        top_k = top_k or settings.retriever.top_k
        threshold = threshold if threshold is not None else settings.retriever.similarity_threshold
        mmr_lambda = mmr_lambda if mmr_lambda is not None else settings.retriever.mmr_lambda

        # Validate mode
        try:
            retrieval_mode = RetrievalMode(mode.lower())
        except ValueError:
            valid = [m.value for m in RetrievalMode]
            raise ValueError(
                f"Unknown retrieval mode: '{mode}'. "
                f"Valid modes: {valid}"
            )

        logger.info(
            f"Retrieving — mode={retrieval_mode.value}, "
            f"query='{query[:60]}...', top_k={top_k}"
        )

        # Build metadata filter
        where_filter = None
        if source_filter:
            where_filter = {"source": source_filter}

        # Dispatch to the appropriate strategy
        if retrieval_mode == RetrievalMode.TOP_K:
            results = self._retrieve_top_k(
                query, top_k, where_filter
            )

        elif retrieval_mode == RetrievalMode.THRESHOLD:
            results = self._retrieve_threshold(
                query, top_k, threshold, where_filter
            )

        elif retrieval_mode == RetrievalMode.MMR:
            results = self._retrieve_mmr(
                query, top_k, mmr_lambda, threshold, where_filter
            )

        else:
            results = []

        logger.info(f"  → {len(results)} result(s) returned")
        return results

    # ── Strategy: Top-K ──────────────────────────────────────────────

    def _retrieve_top_k(
        self,
        query: str,
        top_k: int,
        where_filter: Optional[dict],
    ) -> list[RetrievalResult]:
        """
        Simple cosine similarity retrieval.

        Returns the K chunks with the highest similarity scores.
        Fast but potentially redundant — multiple results may come
        from the same section of the document.
        """
        raw_results = self._store.query(
            query_text=query,
            n_results=top_k,
            where_filter=where_filter,
        )

        return self._parse_chroma_results(raw_results)

    # ── Strategy: Threshold ──────────────────────────────────────────

    def _retrieve_threshold(
        self,
        query: str,
        top_k: int,
        threshold: float,
        where_filter: Optional[dict],
    ) -> list[RetrievalResult]:
        """
        Retrieve all chunks above a minimum similarity score.

        Fetches more candidates than top_k, then filters by score.
        Useful when you don't know how many relevant chunks exist —
        a broad query might match 20 chunks, while a narrow query
        might only match 2.
        """
        # Fetch more candidates to ensure we find all above threshold
        candidate_count = min(
            settings.retriever.mmr_candidate_count,
            self._store.count,
        )
        if candidate_count == 0:
            return []

        raw_results = self._store.query(
            query_text=query,
            n_results=candidate_count,
            where_filter=where_filter,
        )

        parsed = self._parse_chroma_results(raw_results)

        # Filter by threshold
        filtered = [r for r in parsed if r.score >= threshold]

        # Re-rank
        for i, result in enumerate(filtered):
            result.rank = i + 1

        return filtered[:top_k]  # Cap at top_k even after filtering

    # ── Strategy: MMR ────────────────────────────────────────────────

    def _retrieve_mmr(
        self,
        query: str,
        top_k: int,
        mmr_lambda: float,
        threshold: float,
        where_filter: Optional[dict],
    ) -> list[RetrievalResult]:
        """
        Maximum Marginal Relevance retrieval.

        Algorithm:
            1. Embed the query.
            2. Fetch a large candidate set (mmr_candidate_count).
            3. Iteratively select results that maximize:
               λ · sim(query, candidate) - (1-λ) · max(sim(candidate, selected))

        This ensures each selected result is both relevant to the query
        AND different from already-selected results.

        The λ parameter controls the trade-off:
            - λ = 1.0: Pure relevance (identical to top_k)
            - λ = 0.5: Equal weight to relevance and diversity
            - λ = 0.0: Pure diversity (ignores query relevance)
            - λ = 0.7 (default): Strong relevance with meaningful diversity

        Args:
            query:        Search query string.
            top_k:        Number of results to select.
            mmr_lambda:   Trade-off parameter λ.
            threshold:    Minimum similarity to consider.
            where_filter: Optional metadata filter.

        Returns:
            MMR-reranked list of RetrievalResult objects.
        """
        # ── Step 1: Embed the query ──────────────────────────────────
        query_embedding = self._engine.embed_query(query)

        # ── Step 2: Fetch candidate set ──────────────────────────────
        candidate_count = min(
            settings.retriever.mmr_candidate_count,
            self._store.count,
        )
        if candidate_count == 0:
            return []

        raw_results = self._store.query_by_embedding(
            query_embedding=query_embedding.tolist(),
            n_results=candidate_count,
            where_filter=where_filter,
        )

        # ── Step 3: Extract candidate data ───────────────────────────
        if not raw_results["ids"] or not raw_results["ids"][0]:
            return []

        candidate_ids = raw_results["ids"][0]
        candidate_texts = raw_results["documents"][0]
        candidate_metadatas = raw_results["metadatas"][0]
        candidate_distances = raw_results["distances"][0]
        candidate_embeddings = raw_results["embeddings"][0]

        # Convert distances to similarities
        # ChromaDB cosine distance = 1 - cosine_similarity
        candidate_similarities = [
            1.0 - dist for dist in candidate_distances
        ]

        # Pre-filter by threshold
        valid_indices = [
            i for i, sim in enumerate(candidate_similarities)
            if sim >= threshold
        ]

        if not valid_indices:
            logger.debug("  MMR: No candidates above threshold")
            return []

        # ── Step 4: MMR iterative selection ──────────────────────────
        candidate_emb_array = np.array([
            candidate_embeddings[i] for i in valid_indices
        ])
        candidate_sims = [candidate_similarities[i] for i in valid_indices]

        # Track which candidates are selected vs. remaining
        selected_indices: list[int] = []  # Indices into valid_indices
        remaining = set(range(len(valid_indices)))

        for _ in range(min(top_k, len(valid_indices))):
            best_score = -float("inf")
            best_idx = -1

            for idx in remaining:
                # ── Relevance component ──────────────────────────────
                relevance = candidate_sims[idx]

                # ── Diversity component ──────────────────────────────
                # Max similarity between this candidate and all
                # already-selected results
                if selected_indices:
                    selected_embs = candidate_emb_array[selected_indices]
                    candidate_emb = candidate_emb_array[idx]

                    # Dot product (vectors are L2-normalized)
                    inter_similarities = np.dot(
                        selected_embs, candidate_emb
                    )
                    max_inter_sim = float(np.max(inter_similarities))
                else:
                    # No selected results yet → no diversity penalty
                    max_inter_sim = 0.0

                # ── MMR score ────────────────────────────────────────
                # λ · relevance - (1-λ) · max_inter_similarity
                mmr_score = (
                    mmr_lambda * relevance
                    - (1.0 - mmr_lambda) * max_inter_sim
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx == -1:
                break

            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        # ── Step 5: Build results ────────────────────────────────────
        results: list[RetrievalResult] = []
        for rank, sel_idx in enumerate(selected_indices, start=1):
            orig_idx = valid_indices[sel_idx]
            results.append(RetrievalResult(
                text=candidate_texts[orig_idx],
                score=candidate_similarities[orig_idx],
                rank=rank,
                metadata=candidate_metadatas[orig_idx],
            ))

        return results

    # ── Result Parsing ───────────────────────────────────────────────

    @staticmethod
    def _parse_chroma_results(
        raw_results: dict,
    ) -> list[RetrievalResult]:
        """
        Convert ChromaDB's raw query response into RetrievalResult objects.

        ChromaDB returns distances (lower = closer). We convert to
        similarity scores (higher = more similar) for consistency.

        ChromaDB cosine distance = 1 - cosine_similarity
        So: similarity = 1 - distance
        """
        results: list[RetrievalResult] = []

        if not raw_results["ids"] or not raw_results["ids"][0]:
            return results

        ids = raw_results["ids"][0]
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]

        for rank, (doc_id, text, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            similarity = 1.0 - dist  # Convert distance → similarity

            results.append(RetrievalResult(
                text=text,
                score=round(similarity, 4),
                rank=rank,
                metadata=meta or {},
            ))

        return results
