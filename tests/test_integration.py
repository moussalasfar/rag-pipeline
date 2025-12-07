import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from langchain.schema import Document
from src.ingestion import DocumentIngester
from src.retrieval import VectorStore, Retriever
from src.generation import AnswerGenerator, RAGPipeline


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.txt"
        sample_path.write_text("Machine learning is AI. " * 50)
        yield tmpdir


def test_end_to_end_pipeline(temp_data_dir):
    ingester = DocumentIngester()
    sample_file = Path(temp_data_dir) / "sample.txt"
    
    text_content = sample_file.read_text()
    chunks = ingester.ingest_text(text_content, source=str(sample_file))
    
    assert len(chunks) > 0, "Should create chunks from text"
    
    with tempfile.TemporaryDirectory() as index_dir:
        index_path = Path(index_dir) / "test_index.faiss"
        vector_store = VectorStore(index_path=str(index_path))
        
        with patch.object(vector_store.embeddings, 'embed_documents') as mock_docs:
            with patch.object(vector_store.embeddings, 'embed_query') as mock_query:
                mock_docs.return_value = [[0.1] * 1536] * len(chunks)
                mock_query.return_value = [0.1] * 1536
                
                vector_store.add_documents(chunks)
                assert len(vector_store.documents) == len(chunks)
                
                retriever = Retriever(vector_store)
                results = retriever.retrieve("machine learning", k=2)
                
                assert len(results) > 0, "Should retrieve documents"
                assert all('content' in r for r in results)
                
                generator = AnswerGenerator()
                
                with patch.object(generator.llm, 'call') as mock_call:
                    from langchain.schema import AIMessage
                    mock_call.return_value = AIMessage(content="Machine learning is a type of artificial intelligence.")
                    
                    answer = generator.generate("What is machine learning?", results)
                    
                    assert isinstance(answer, str)
                    assert len(answer) > 0


def test_pipeline_with_multiple_documents():
    docs = [
        Document(
            page_content=f"Document {i}: " + "content " * 50,
            metadata={'source': f'doc{i}.txt', 'chunk_id': j}
        )
        for i in range(2)
        for j in range(3)
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "index.faiss"
        vector_store = VectorStore(index_path=str(index_path))
        
        with patch.object(vector_store.embeddings, 'embed_documents') as mock_docs:
            with patch.object(vector_store.embeddings, 'embed_query') as mock_query:
                mock_docs.return_value = [[0.1] * 1536] * len(docs)
                mock_query.return_value = [0.1] * 1536
                
                vector_store.add_documents(docs)
                
                assert len(vector_store.documents) == len(docs)
                
                retriever = Retriever(vector_store)
                results = retriever.retrieve("test query", k=3)
                
                assert len(results) <= 3
                assert all('source' in r for r in results)
