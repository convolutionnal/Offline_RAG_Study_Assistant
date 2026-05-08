import pytest
from unittest.mock import MagicMock
import numpy as np

from app.core.chunker import SemanticChunker, _build_breadcrumb
from app.core.document_loader import Document
from app.core.embeddings import EmbeddingEngine

@pytest.fixture
def mock_embedding_engine():
    engine = MagicMock(spec=EmbeddingEngine)
    # Default behavior: return random normalized embeddings
    def mock_embed_texts(texts):
        # Return dummy embeddings (normalized)
        embeddings = np.random.rand(len(texts), 768)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    engine.embed_texts.side_effect = mock_embed_texts
    return engine

def test_header_aware_split(mock_embedding_engine):
    """1. Test Markdown Header Splitting (Pass 1)"""
    chunker = SemanticChunker(
        embedding_engine=mock_embedding_engine,
        min_chunk_size=10,
        max_chunk_size=1000
    )
    
    md_text = "# Chapter 1\nThis is chapter 1.\n## Section A\nThis is section A."
    doc = Document(page_content=md_text, metadata={"source": "test.pdf", "page": 1})
    
    chunks = chunker.chunk([doc])
    
    # We expect 2 chunks if split by headers
    assert len(chunks) == 2
    
    # Check first chunk metadata
    assert chunks[0].metadata["section"] == "Chapter 1"
    assert chunks[0].metadata.get("subsection", "") == ""
    assert "Chapter 1" in chunks[0].text
    
    # Check second chunk metadata
    assert chunks[1].metadata["section"] == "Chapter 1"
    assert chunks[1].metadata["subsection"] == "Section A"
    assert "Chapter 1 > Section A" in chunks[1].text

def test_semantic_split_oversized(mock_embedding_engine):
    """2. Test Semantic Splitting (Pass 2)"""
    chunker = SemanticChunker(
        embedding_engine=mock_embedding_engine,
        min_chunk_size=10,
        max_chunk_size=100,
        breakpoint_k=0.5
    )
    
    # 4 sentences
    text = "Topic A sentence 1. Topic A sentence 2. Topic B sentence 1. Topic B sentence 2."
    
    # We mock embed_texts to return specific vectors
    def custom_embed(texts):
        # Topic A vectors (similar)
        vec_A1 = np.array([1.0, 0.0])
        vec_A2 = np.array([0.9, 0.1])
        # Topic B vectors (similar to each other, different from A)
        vec_B1 = np.array([0.0, 1.0])
        vec_B2 = np.array([0.1, 0.9])
        
        embs = np.array([vec_A1, vec_A2, vec_B1, vec_B2])
        # Normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norms

    mock_embedding_engine.embed_texts.side_effect = custom_embed
    
    chunks = chunker._semantic_split_text(text)
    
    # Expect 2 chunks because of the semantic shift between sentence 2 and 3
    assert len(chunks) == 2
    assert "Topic A" in chunks[0]
    assert "Topic B" in chunks[1]

def test_size_guardrails(mock_embedding_engine):
    """3. Test Size Guardrails"""
    chunker = SemanticChunker(
        embedding_engine=mock_embedding_engine,
        min_chunk_size=50,
        max_chunk_size=100
    )
    
    # Scenario A (Min Size): Merge small chunks
    raw_texts = ["Tiny chunk.", "Another tiny one."]
    merged = chunker._apply_size_guardrails(raw_texts)
    assert len(merged) == 1
    assert merged[0] == "Tiny chunk. Another tiny one."
    
    # Scenario B (Max Size): Split oversized chunks
    long_str_sentences = "This is a sentence. " * 10 # 200 chars
    split = chunker._apply_size_guardrails([long_str_sentences])
    assert len(split) > 1
    for s in split:
        assert len(s) <= 100

def test_breadcrumb_injection():
    """4. Test Breadcrumb Injection"""
    metadata = {"section": "OS", "subsection": "Memory"}
    breadcrumb = _build_breadcrumb(metadata)
    assert breadcrumb == "OS > Memory"
    
    # Empty case
    assert _build_breadcrumb({}) == ""

def test_generate_chunk_id(mock_embedding_engine):
    """5. Test Deterministic Chunk IDs"""
    chunker = SemanticChunker(embedding_engine=mock_embedding_engine)
    
    id1 = chunker._generate_chunk_id("file.pdf", 0)
    id2 = chunker._generate_chunk_id("file.pdf", 0)
    id3 = chunker._generate_chunk_id("file.pdf", 1)
    
    # Deterministic (same inputs produce same hash)
    assert id1 == id2
    # Unique by index
    assert id1 != id3
