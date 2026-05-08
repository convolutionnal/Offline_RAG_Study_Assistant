"""
Quick end-to-end test of the fixed pipeline.
Uses the PDFs in data/uploads/ directory.
"""
import sys
import time

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from pathlib import Path
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.config import settings
from app.core.document_loader import PDFDocumentLoader
from app.core.text_cleaner import TextCleaner
from app.core.embeddings import EmbeddingEngine
from app.core.chunker import SemanticChunker
from app.core.vector_store import ChromaVectorStore
from app.core.retriever import AdvancedRetriever

PDF_DIR = _BACKEND_DIR / "data" / "uploads"

def main():
    settings.paths.ensure_directories()

    print("=" * 60)
    print("  STAGE 1: PDF Loading (pymupdf4llm)")
    print("=" * 60)
    t0 = time.perf_counter()
    loader = PDFDocumentLoader(PDF_DIR)
    raw_docs = loader.load_all()
    t1 = time.perf_counter()
    sources = {d.metadata['source'] for d in raw_docs}
    print(f"  PDFs found:       {len(sources)}")
    print(f"  Pages extracted:  {len(raw_docs)}")
    print(f"  Total chars:      {sum(len(d.page_content) for d in raw_docs):,}")
    print(f"  Time:             {t1-t0:.2f}s")
    # Show a snippet of the first page's Markdown
    if raw_docs:
        print(f"\n  --- First page preview (first 300 chars) ---")
        print(f"  {raw_docs[0].page_content[:300]}")
    print()

    print("=" * 60)
    print("  STAGE 2: Text Cleaning")
    print("=" * 60)
    t0 = time.perf_counter()
    cleaner = TextCleaner()
    cleaned = cleaner.clean(raw_docs)
    t1 = time.perf_counter()
    raw_chars = sum(len(d.page_content) for d in raw_docs)
    clean_chars = sum(len(d.page_content) for d in cleaned)
    reduction = (1 - clean_chars / raw_chars) * 100 if raw_chars > 0 else 0
    print(f"  Pages retained:   {len(cleaned)}/{len(raw_docs)}")
    print(f"  Characters:       {raw_chars:,} -> {clean_chars:,}")
    print(f"  Noise removed:    {reduction:.1f}%")
    print(f"  Time:             {t1-t0:.2f}s")
    print()

    print("=" * 60)
    print("  STAGE 3: Embedding Model Init")
    print("=" * 60)
    t0 = time.perf_counter()
    engine = EmbeddingEngine()
    _ = engine.embed_query("warmup query")
    t1 = time.perf_counter()
    print(f"  Model:            {engine._model_name}")
    print(f"  Device:           {engine._device}")
    print(f"  Dimensions:       {settings.embedding.embedding_dimension}")
    print(f"  Load time:        {t1-t0:.2f}s")
    print()

    print("=" * 60)
    print("  STAGE 4: Hybrid Chunking (Header + Semantic)")
    print("=" * 60)
    t0 = time.perf_counter()
    chunker = SemanticChunker(embedding_engine=engine)
    chunks = chunker.chunk(cleaned)
    t1 = time.perf_counter()
    if chunks:
        sizes = [len(c.text) for c in chunks]
        print(f"  Total chunks:     {len(chunks)}")
        print(f"  Min size:         {min(sizes)} chars")
        print(f"  Avg size:         {sum(sizes)//len(sizes)} chars")
        print(f"  Max size:         {max(sizes)} chars")
        print(f"  Time:             {t1-t0:.2f}s")
        # Show a couple example chunks
        print(f"\n  --- Example chunk [0] ---")
        print(f"  Section: {chunks[0].metadata.get('section','')}")
        print(f"  {chunks[0].text[:200]}...")
        if len(chunks) > 5:
            print(f"\n  --- Example chunk [5] ---")
            print(f"  Section: {chunks[5].metadata.get('section','')}")
            print(f"  {chunks[5].text[:200]}...")
    print()

    print("=" * 60)
    print("  STAGE 5: ChromaDB Vector Storage")
    print("=" * 60)
    t0 = time.perf_counter()
    store = ChromaVectorStore(engine)
    store.delete_collection()  # Fresh start
    store = ChromaVectorStore(engine)  # Re-create
    store.add_chunks(chunks)
    t1 = time.perf_counter()
    stats = store.get_collection_stats()
    print(f"  Collection:       {stats['collection_name']}")
    print(f"  Chunks stored:    {stats['total_chunks']}")
    print(f"  Time:             {t1-t0:.2f}s")
    print()

    print("=" * 60)
    print("  STAGE 6: Retrieval Tests")
    print("=" * 60)
    retriever = AdvancedRetriever(store, engine)

    queries = [
        "What are word embeddings?",
        "How does attention mechanism work in transformers?",
    ]

    for q in queries:
        print(f"\n  Query: \"{q}\"")
        results = retriever.retrieve(q, mode="top_k", top_k=3)
        for r in results:
            pages = f"p{r.metadata.get('page_start','?')}-{r.metadata.get('page_end','?')}"
            preview = r.text[:100].replace('\n', ' ')
            print(f"    #{r.rank} [score={r.score:.4f}] [{pages}] {preview}...")
        print()

    print("=" * 60)
    print("  ALL STAGES COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
