#!/bin/bash
# Setup script for RAG pipeline

set -e

echo "🚀 Setting up RAG Pipeline..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate || . venv/Scripts/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/documents data/index

# Verify setup
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Add your OpenAI API key to .env file"
echo "3. Place PDF files in data/documents/"
echo "4. Run: python src/cli.py ingest data/documents"
echo "5. Query: python src/cli.py query 'Your question here'"
