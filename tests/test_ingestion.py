"""Tests for document ingestion."""

import pytest
from pathlib import Path
from src.ingestion import DocumentIngester


@pytest.fixture
def ingester():
    """Create ingester instance."""
    return DocumentIngester(chunk_size=500, chunk_overlap=50)


def test_ingester_initialization(ingester):
    """Test ingester initialization."""
    assert ingester.chunk_size == 500
    assert ingester.chunk_overlap == 50
    assert ingester.splitter is not None


def test_ingest_text(ingester):
    """Test text ingestion and chunking."""
    text = "This is a test document. " * 100  # Create longer text
    chunks = ingester.ingest_text(text, source="test_source")
    
    assert len(chunks) > 0
    assert all(len(c.page_content) <= 550 for c in chunks)  # chunk_size + overlap
    assert all(c.metadata['source'] == 'test_source' for c in chunks)
    assert all('chunk_id' in c.metadata for c in chunks)


def test_chunk_metadata(ingester):
    """Test that chunk metadata is properly set."""
    text = "Test chunk " * 100
    chunks = ingester.ingest_text(text)
    
    for i, chunk in enumerate(chunks):
        assert chunk.metadata['chunk_id'] == i
        assert chunk.metadata['chunk_size'] > 0
        assert 'source' in chunk.metadata


def test_empty_text(ingester):
    """Test handling of empty text."""
    chunks = ingester.ingest_text("")
    assert len(chunks) == 0


def test_small_text(ingester):
    """Test handling of text smaller than chunk size."""
    text = "Short text"
    chunks = ingester.ingest_text(text)
    
    assert len(chunks) == 1
    assert chunks[0].page_content == text
