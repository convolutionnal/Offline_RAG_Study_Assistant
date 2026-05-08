"""
pipeline_runner.py — End-to-End Integration Test
==================================================

Proves the entire RAG pipeline works by executing every stage
sequentially: PDF loading -> cleaning -> semantic chunking ->
embedding & storage -> retrieval (all 3 modes).

Outputs a rich, formatted console report with timing, statistics,
and retrieval results.

Usage:
    cd backend
    python pipeline_runner.py

    # Force re-ingestion (delete existing ChromaDB data):
    python pipeline_runner.py --fresh

Requirements:
    - At least one PDF in the sample_docs/ directory.
    - All dependencies installed (pip install -r requirements.txt).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Ensure backend/ is on the Python path ────────────────────────────
# This allows `from app.xxx import yyy` to work when running
# this script directly from the backend/ directory.
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.config import settings
from app.core.chunker import SemanticChunker
from app.core.document_loader import PDFDocumentLoader
from app.core.embeddings import EmbeddingEngine
from app.core.retriever import AdvancedRetriever
from app.core.text_cleaner import TextCleaner
from app.core.vector_store import ChromaVectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Rich Console Helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_console():
    """Get a Rich console instance (with plain fallback)."""
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


def _print_header(console, title: str) -> None:
    """Print a styled section header."""
    if console:
        from rich.panel import Panel
        console.print()
        console.print(Panel(
            f"[bold white]{title}[/bold white]",
            border_style="cyan",
            expand=False,
        ))
    else:
        print(f"\n{'='*60}\n  {title}\n{'='*60}")


def _print_stats_table(console, title: str, rows: list[tuple]) -> None:
    """Print a key-value stats table."""
    if console:
        from rich.table import Table
        table = Table(title=title, show_header=False, border_style="dim")
        table.add_column("Metric", style="cyan", min_width=25)
        table.add_column("Value", style="white")
        for key, value in rows:
            table.add_row(key, str(value))
        console.print(table)
    else:
        print(f"\n  {title}")
        for key, value in rows:
            print(f"    {key}: {value}")


def _print_results_table(console, title: str, results) -> None:
    """Print retrieval results in a formatted table."""
    if console:
        from rich.table import Table
        table = Table(title=title, border_style="green")
        table.add_column("#", style="bold", width=3)
        table.add_column("Score", style="yellow", width=8)
        table.add_column("Source", style="cyan", width=18)
        table.add_column("Pages", width=8)
        table.add_column("Preview", style="white", max_width=55)
        for r in results:
            pages = f"{r.metadata.get('page_start', '?')}-{r.metadata.get('page_end', '?')}"
            preview = r.text[:80].replace("\n", " ")
            table.add_row(
                str(r.rank),
                f"{r.score:.4f}",
                r.metadata.get("source", "?"),
                pages,
                preview + "...",
            )
        console.print(table)
    else:
        print(f"\n  {title}")
        for r in results:
            print(f"    #{r.rank} [score={r.score:.4f}] {r.text[:80]}...")


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Stages
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(fresh: bool = False) -> None:
    """
    Execute the full RAG pipeline from PDF to retrieval.

    Args:
        fresh: If True, delete existing ChromaDB data before ingestion.
    """
    console = _get_console()
    timings: dict[str, float] = {}

    if console:
        from rich.panel import Panel
        console.print(Panel(
            "[bold cyan]Offline RAG Study Assistant[/bold cyan]\n"
            "[dim]End-to-End Pipeline Integration Test[/dim]",
            border_style="bright_blue",
            expand=False,
        ))

    # ── Ensure directories exist ─────────────────────────────────────
    settings.paths.ensure_directories()

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1: Document Loading
    # ══════════════════════════════════════════════════════════════════
    _print_header(console, "Stage 1: Document Loading")
    t0 = time.perf_counter()

    try:
        loader = PDFDocumentLoader(settings.paths.pdf_input_dir)
        raw_documents = loader.load_all()
    except FileNotFoundError as e:
        logger.error(f"Cannot proceed: {e}")
        logger.error(
            "Please add at least one PDF to: "
            f"{settings.paths.pdf_input_dir}"
        )
        return

    timings["loading"] = time.perf_counter() - t0

    _print_stats_table(console, "Loading Results", [
        ("PDFs found", len({d.metadata['source'] for d in raw_documents})),
        ("Pages extracted", len(raw_documents)),
        ("Total characters", f"{sum(len(d.page_content) for d in raw_documents):,}"),
        ("Time", f"{timings['loading']:.2f}s"),
    ])

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2: Text Cleaning
    # ══════════════════════════════════════════════════════════════════
    _print_header(console, "Stage 2: Text Cleaning")
    t0 = time.perf_counter()

    cleaner = TextCleaner()
    cleaned_documents = cleaner.clean(raw_documents)

    timings["cleaning"] = time.perf_counter() - t0

    raw_chars = sum(len(d.page_content) for d in raw_documents)
    clean_chars = sum(len(d.page_content) for d in cleaned_documents)
    reduction = (1 - clean_chars / raw_chars) * 100 if raw_chars > 0 else 0

    _print_stats_table(console, "Cleaning Results", [
        ("Pages retained", f"{len(cleaned_documents)}/{len(raw_documents)}"),
        ("Characters", f"{raw_chars:,} -> {clean_chars:,}"),
        ("Noise removed", f"{reduction:.1f}%"),
        ("Time", f"{timings['cleaning']:.2f}s"),
    ])

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3: Embedding Engine Initialization
    # ══════════════════════════════════════════════════════════════════
    _print_header(console, "Stage 3: Loading Embedding Model")
    t0 = time.perf_counter()

    embedding_engine = EmbeddingEngine()
    # Force model load now (not lazily during chunking)
    _ = embedding_engine.embed_query("warmup")

    timings["model_load"] = time.perf_counter() - t0

    _print_stats_table(console, "Embedding Model", [
        ("Model", settings.embedding.model_name),
        ("Dimensions", settings.embedding.embedding_dimension),
        ("Device", settings.embedding.device),
        ("Normalized", settings.embedding.normalize),
        ("Load time", f"{timings['model_load']:.2f}s"),
    ])

    # ══════════════════════════════════════════════════════════════════
    # STAGE 4: Semantic Chunking
    # ══════════════════════════════════════════════════════════════════
    _print_header(console, "Stage 4: Semantic Chunking")
    t0 = time.perf_counter()

    chunker = SemanticChunker(embedding_engine=embedding_engine)
    chunks = chunker.chunk(cleaned_documents)

    timings["chunking"] = time.perf_counter() - t0

    if chunks:
        sizes = [len(c.text) for c in chunks]
        import numpy as np
        _print_stats_table(console, "Chunking Results", [
            ("Total chunks", len(chunks)),
            ("Sensitivity (k)", settings.chunker.breakpoint_sensitivity_k),
            ("Min chunk size", f"{min(sizes)} chars"),
            ("Avg chunk size", f"{int(np.mean(sizes))} chars"),
            ("Max chunk size", f"{max(sizes)} chars"),
            ("Time", f"{timings['chunking']:.2f}s"),
        ])

    # ══════════════════════════════════════════════════════════════════
    # STAGE 5: Vector Storage (ChromaDB)
    # ══════════════════════════════════════════════════════════════════
    _print_header(console, "Stage 5: Vector Storage (ChromaDB)")
    t0 = time.perf_counter()

    vector_store = ChromaVectorStore(embedding_engine)

    if fresh:
        logger.info("--fresh flag: deleting existing collection...")
        vector_store.delete_collection()

    # Check if already ingested (idempotent)
    existing_count = vector_store.count
    if existing_count > 0 and not fresh:
        logger.info(
            f"Collection already has {existing_count} chunks. "
            f"Upserting (idempotent)..."
        )

    vector_store.add_chunks(chunks)

    timings["storage"] = time.perf_counter() - t0

    stats = vector_store.get_collection_stats()
    _print_stats_table(console, "Storage Results", [
        ("Collection", stats["collection_name"]),
        ("Total chunks stored", stats["total_chunks"]),
        ("Distance metric", stats["distance_metric"]),
        ("Persist directory", stats["persist_dir"]),
        ("Time", f"{timings['storage']:.2f}s"),
    ])

    # ══════════════════════════════════════════════════════════════════
    # STAGE 6: Retrieval Tests
    # ══════════════════════════════════════════════════════════════════
    _print_header(console, "Stage 6: Retrieval Tests")

    retriever = AdvancedRetriever(vector_store, embedding_engine)

    # Define test queries
    test_queries = [
        "What is virtual memory and how does paging work?",
        "Explain the difference between process and thread",
        "What are the ACID properties in database systems?",
    ]

    for query in test_queries:
        if console:
            console.print(f"\n[bold yellow]Query:[/bold yellow] \"{query}\"")

        # ── Test 1: Top-K ────────────────────────────────────────────
        t0 = time.perf_counter()
        top_k_results = retriever.retrieve(query, mode="top_k", top_k=3)
        t_topk = time.perf_counter() - t0

        if top_k_results:
            _print_results_table(
                console,
                f"Top-K Results ({t_topk:.3f}s)",
                top_k_results,
            )

        # ── Test 2: MMR ──────────────────────────────────────────────
        t0 = time.perf_counter()
        mmr_results = retriever.retrieve(query, mode="mmr", top_k=3)
        t_mmr = time.perf_counter() - t0

        if mmr_results:
            _print_results_table(
                console,
                f"MMR Results λ={settings.retriever.mmr_lambda} ({t_mmr:.3f}s)",
                mmr_results,
            )

        # ── Test 3: Threshold ────────────────────────────────────────
        t0 = time.perf_counter()
        thresh_results = retriever.retrieve(
            query, mode="threshold", top_k=10, threshold=0.4
        )
        t_thresh = time.perf_counter() - t0

        if thresh_results:
            _print_results_table(
                console,
                f"Threshold≥0.4 Results ({t_thresh:.3f}s)",
                thresh_results,
            )
        elif console:
            console.print(
                "  [dim]No results above threshold 0.4[/dim]"
            )

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    _print_header(console, "Pipeline Summary")

    total_time = sum(timings.values())
    _print_stats_table(console, "Timing Breakdown", [
        ("PDF Loading", f"{timings['loading']:.2f}s"),
        ("Text Cleaning", f"{timings['cleaning']:.2f}s"),
        ("Model Loading", f"{timings['model_load']:.2f}s"),
        ("Semantic Chunking", f"{timings['chunking']:.2f}s"),
        ("Vector Storage", f"{timings['storage']:.2f}s"),
        ("-" * 20, "-" * 10),
        ("Total Pipeline", f"{total_time:.2f}s"),
    ])

    if console:
        from rich.panel import Panel
        console.print(Panel(
            "[bold green]Pipeline completed successfully![/bold green]\n"
            f"[dim]{len(chunks)} chunks from "
            f"{len({c.metadata['source'] for c in chunks})} PDF(s) "
            f"indexed in ChromaDB[/dim]",
            border_style="green",
            expand=False,
        ))


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full RAG pipeline from PDF to retrieval."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing ChromaDB data and re-ingest from scratch.",
    )
    args = parser.parse_args()

    run_pipeline(fresh=args.fresh)
