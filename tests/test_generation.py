"""Tests for answer generation."""

import pytest
from unittest.mock import Mock, patch
from langchain.schema import AIMessage
from src.generation import AnswerGenerator, RAGPipeline


@pytest.fixture
def answer_generator():
    """Create answer generator instance."""
    return AnswerGenerator(model="gpt-3.5-turbo", temperature=0.7)


@pytest.fixture
def sample_context():
    """Create sample context for testing."""
    return [
        {
            'content': 'Machine learning is a branch of artificial intelligence.',
            'source': 'doc1.pdf',
            'score': 0.95
        },
        {
            'content': 'It focuses on the development of algorithms.',
            'source': 'doc2.pdf',
            'score': 0.87
        }
    ]


def test_answer_generator_initialization(answer_generator):
    """Test answer generator initialization."""
    assert answer_generator.llm is not None


def test_format_context_single_doc(answer_generator):
    """Test context formatting with single document."""
    context = [
        {'content': 'Test content', 'source': 'test.pdf', 'score': 0.9}
    ]
    
    formatted = answer_generator._format_context(context)
    
    assert 'Document 1' in formatted
    assert 'test.pdf' in formatted
    assert 'Test content' in formatted


def test_format_context_multiple_docs(answer_generator, sample_context):
    """Test context formatting with multiple documents."""
    formatted = answer_generator._format_context(sample_context)
    
    assert 'Document 1' in formatted
    assert 'Document 2' in formatted
    assert 'doc1.pdf' in formatted
    assert 'doc2.pdf' in formatted


def test_generate_answer(answer_generator, sample_context):
    """Test answer generation."""
    with patch.object(answer_generator.llm, 'call') as mock_call:
        mock_call.return_value = AIMessage(content="Machine learning is a branch of AI.")
        
        answer = answer_generator.generate("What is machine learning?", sample_context)
        
        assert isinstance(answer, str)
        assert len(answer) > 0


def test_rag_pipeline_initialization(answer_generator):
    """Test RAG pipeline initialization."""
    mock_retriever = Mock()
    pipeline = RAGPipeline(mock_retriever, answer_generator)
    
    assert pipeline.retriever == mock_retriever
    assert pipeline.generator == answer_generator


def test_rag_pipeline_answer(answer_generator, sample_context):
    """Test RAG pipeline answer generation."""
    mock_retriever = Mock()
    mock_retriever.retrieve.return_value = sample_context
    
    pipeline = RAGPipeline(mock_retriever, answer_generator)
    
    with patch.object(answer_generator.llm, 'call') as mock_call:
        mock_call.return_value = AIMessage(content="Answer about ML")
        
        result = pipeline.answer("What is ML?", k=2)
    
    assert 'query' in result
    assert 'retrieved_documents' in result
    assert 'answer' in result
    assert 'num_sources' in result
    assert result['num_sources'] == len(sample_context)


def test_empty_context(answer_generator):
    """Test answer generation with empty context."""
    with patch.object(answer_generator.llm, 'call') as mock_call:
        mock_call.return_value = AIMessage(content="I don't have information to answer this.")
        
        answer = answer_generator.generate("Test query", [])
        
        assert isinstance(answer, str)
