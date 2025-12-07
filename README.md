# RAG Pipeline - Quorium AI Engineer Challenge

A clean, production-ready Retrieval-Augmented Generation (RAG) pipeline with thoughtful document ingestion, semantic search, and answer generation.

## Features

- **Smart Document Ingestion**: PDF parsing with chunking and metadata extraction
- **FAISS Vector Store**: Fast, scalable similarity search on embeddings
- **Semantic Retrieval**: Retrieve relevant chunks based on queries
- **Answer Generation**: LLM-powered responses grounded in retrieved context
- **Comprehensive Tests**: Unit and integration tests for all critical paths

## Quick Start

### 1. Setup Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Add Your OpenAI API Key

```bash
# Create .env file
echo OPENAI_API_KEY=your_key_here > .env
```

### 3. Run the Pipeline

```bash
# Ingest documents
python src/cli.py ingest data/documents

# Query the knowledge base
python src/cli.py query "What is your question?"
```

## Project Structure

```
stage_proj/
├── src/
│   ├── __init__.py
│   ├── ingestion.py      # Document loading & chunking
│   ├── retrieval.py      # Vector store & semantic search
│   ├── generation.py     # LLM-powered answer generation
│   ├── cli.py            # Command-line interface
│   └── utils.py          # Helper functions
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── data/
│   ├── documents/        # Input PDFs
│   ├── index/            # FAISS vector store (auto-created)
│   └── sample.txt        # Sample document
├── requirements.txt
├── README.md
└── .env (create this)
```

## Architecture Decisions

### 1. Document Chunking
- **Strategy**: Recursive character split with 500-char chunks, 50-char overlap
- **Rationale**: Balances context preservation with retrieval precision
- **Tradeoff**: Larger chunks preserve context but reduce retrieval granularity

### 2. Embeddings & Vector Store
- **Choice**: OpenAI embeddings + FAISS
- **Rationale**: High-quality embeddings, fast CPU-based search, easy to scale to GPU
- **Tradeoff**: Requires OpenAI API; could use open-source embeddings for offline use

### 3. Retrieval Strategy
- **Method**: Top-k similarity search (k=4 by default)
- **Rationale**: Small k improves LLM context window efficiency
- **Tradeoff**: Fewer documents reduce coverage but improve answer quality

### 4. Answer Generation
- **Approach**: Few-shot prompting with retrieved context
- **Rationale**: Simple, interpretable, minimal hallucination
- **Tradeoff**: No fine-tuning; could improve with few-shot examples

## Running Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_ingestion.py -v

# Run with coverage
pytest --cov=src tests/
```

## Docker Setup (Optional)

### Build and run with Docker:

```bash
# Build the image
docker build -t rag-pipeline .

# Run ingestion
docker run --rm -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/data/documents:/app/data/documents \
  -v $(pwd)/data/index:/app/data/index \
  rag-pipeline python src/cli.py ingest data/documents

# Or use docker-compose
docker-compose run rag-pipeline python src/cli.py ingest data/documents
```

## Validation

Critical paths tested:
- ✅ PDF parsing and chunking
- ✅ Embedding generation and storage
- ✅ Similarity search retrieval
- ✅ Answer generation with context
- ✅ End-to-end ingestion → query pipeline

Run validation script:
```bash
python validate.py
```

## Next Steps / Improvements

1. **Hybrid Search**: Combine vector + BM25 retrieval for robustness
2. **Re-ranking**: Add cross-encoder re-ranking for top-k results
3. **Query Expansion**: Use LLM to expand queries for better retrieval
4. **Feedback Loop**: Collect relevance feedback to improve retrieval
5. **Multi-modal**: Support images and tables in documents
6. **Caching**: Redis-backed embedding cache for repeated queries
7. **Monitoring**: Log retrieval quality and answer relevance scores
8. **Database Backend**: Replace FAISS with Postgres + pgvector for production

## Development Notes

- All code follows PEP 8 style guidelines
- Type hints used throughout for clarity
- Async support ready for scalability
- Designed for easy OpenAI key rotation and embedding model swaps

## License

MIT
