import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from langchain.schema import Document
from src.retrieval import VectorStore, Retriever


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="Machine learning is a subset of AI", metadata={'source': 'doc1.pdf', 'chunk_id': 0}),
        Document(page_content="Neural networks are inspired by biological neurons", metadata={'source': 'doc2.pdf', 'chunk_id': 0}),
        Document(page_content="Deep learning uses multiple layers of neural networks", metadata={'source': 'doc2.pdf', 'chunk_id': 1}),
    ]


@pytest.fixture
def vector_store(sample_docs):
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index.faiss"
        store = VectorStore(index_path=str(index_path))
        
        with patch.object(store.embeddings, 'embed_documents') as mock_embed_docs:
            with patch.object(store.embeddings, 'embed_query') as mock_embed_query:
                mock_embed_docs.return_value = [[0.1, 0.2, 0.3]] * len(sample_docs)
                mock_embed_query.return_value = [0.1, 0.2, 0.3]
                
                store.add_documents(sample_docs)
                
                yield store


def test_vector_store_initialization():
    store = VectorStore()
    assert store.embeddings is not None
    assert store.index is None


def test_vector_store_add_documents(vector_store, sample_docs):
    assert vector_store.index is not None
    assert len(vector_store.documents) == len(sample_docs)


def test_retriever_initialization(vector_store):
    retriever = Retriever(vector_store)
    assert retriever.vector_store == vector_store


def test_retriever_retrieve(vector_store):
    retriever = Retriever(vector_store)
    
    with patch.object(vector_store.embeddings, 'embed_query') as mock_embed:
        mock_embed.return_value = [0.1, 0.2, 0.3]
        results = retriever.retrieve("machine learning", k=2)
    
    assert len(results) <= 2
    assert all('content' in r for r in results)
    assert all('source' in r for r in results)
    assert all('score' in r for r in results)


def test_search_without_index():
    store = VectorStore()
    
    with pytest.raises(RuntimeError):
        store.search("test query")


def test_retrieval_quality(vector_store):
    retriever = Retriever(vector_store)
    
    with patch.object(vector_store.embeddings, 'embed_query') as mock_embed:
        mock_embed.return_value = [0.1, 0.2, 0.3]
        results = retriever.retrieve("neural networks", k=3)
    
    assert all(0 <= r['score'] <= 1 for r in results)
