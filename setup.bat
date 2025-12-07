@echo off
REM Setup script for RAG pipeline (Windows)

echo 🚀 Setting up RAG Pipeline...

REM Create virtual environment
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Create data directories
echo 📁 Creating data directories...
if not exist "data\documents" mkdir data\documents
if not exist "data\index" mkdir data\index

echo ✅ Setup complete!
echo.
echo Next steps:
echo 1. Activate the environment: venv\Scripts\activate
echo 2. Add your OpenAI API key to .env file
echo 3. Place PDF files in data/documents/
echo 4. Run: python src/cli.py ingest data/documents
echo 5. Query: python src/cli.py query "Your question here"

pause
