## Quick Start Guide

### Prerequisites
- Python 3.8+
- OpenAI API key (get one from https://platform.openai.com/api-keys)
- Git

### Installation & Setup

1. **Clone and navigate to the project:**
   ```bash
   cd c:\Users\acer\OneDrive\Bureau\stage_proj
   ```

2. **Run the setup script:**
   ```bash
   # Windows
   setup.bat
   
   # macOS/Linux
   bash setup.sh
   ```

3. **Configure OpenAI API key:**
   ```bash
   # Copy the environment template
   copy .env.example .env
   # Or on macOS/Linux
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=sk-...
   ```

### Running the Pipeline

#### Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

#### Ingest Documents
```bash
# Place your PDF files in data/documents/
# Then ingest them:
python src/cli.py ingest data/documents
```

#### Query the Knowledge Base
```bash
python src/cli.py query "What is your question?"
```

### Running Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_ingestion.py -v

# Run with coverage report
pytest --cov=src tests/
```

### Validation

Check the pipeline is properly set up:
```bash
python validate.py
```

### Project Structure

```
stage_proj/
├── README.md              # Full documentation
├── QUICKSTART.md          # This file
├── requirements.txt       # Python dependencies
├── validate.py            # Validation script
├── setup.bat/setup.sh     # Setup scripts
│
├── src/                   # Source code
│   ├── ingestion.py       # Document parsing & chunking
│   ├── retrieval.py       # Vector store & search
│   ├── generation.py      # LLM answer generation
│   ├── cli.py             # Command-line interface
│   └── utils.py           # Helper functions
│
├── tests/                 # Test suite
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_integration.py
│
└── data/                  # Data directory
    ├── documents/         # Input PDFs (create this)
    ├── index/             # Vector store index (auto-created)
    └── sample.txt         # Sample document
```

### Key Components

**DocumentIngester** - Loads PDFs and chunks them into 500-character segments

**VectorStore** - Stores embeddings in FAISS for fast semantic search

**Retriever** - Retrieves top-k most relevant documents for a query

**AnswerGenerator** - Uses GPT-3.5-turbo to generate answers grounded in context

**RAGPipeline** - Orchestrates the full flow: query → retrieve → answer

### Example Workflow

```bash
# 1. Activate environment
venv\Scripts\activate

# 2. Ingest documents (make sure PDFs are in data/documents/)
python src/cli.py ingest data/documents

# 3. Query the knowledge base
python src/cli.py query "What are the main benefits?"

# Output will show:
# - Retrieved documents with relevance scores
# - Generated answer grounded in the documents
```

### Troubleshooting

**ImportError: No module named 'langchain'**
- Run `pip install -r requirements.txt` after activating the venv

**No API key error**
- Make sure .env file exists and has OPENAI_API_KEY=sk-...

**No documents found**
- Place PDF files in data/documents/ folder
- Run: `python src/cli.py ingest data/documents`

### Next Steps

- ✅ Review README.md for architecture decisions
- ✅ Run tests: `pytest -v`
- ✅ Experiment with different queries
- ✅ Add more documents to improve retrieval

### Support

For questions or issues, refer to the detailed README.md
