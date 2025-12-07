"""
Validation script to verify the RAG pipeline structure and components.
Run this to ensure all critical paths are properly implemented.
"""

import os
import sys
from pathlib import Path


def check_file_exists(path, description):
    """Check if a file exists and print status."""
    if Path(path).exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} - NOT FOUND")
        return False


def check_directory_exists(path, description):
    """Check if a directory exists and print status."""
    if Path(path).exists() and Path(path).is_dir():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} - NOT FOUND")
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("RAG PIPELINE VALIDATION")
    print("=" * 60)
    print()
    
    base_path = Path(__file__).parent
    all_checks_passed = True
    
    # Check essential files
    print("📄 Checking Essential Files...")
    print("-" * 60)
    essential_files = {
        "README.md": "README documentation",
        "requirements.txt": "Python dependencies",
        ".gitignore": "Git ignore file",
        ".env.example": "Environment template",
    }
    
    for file, desc in essential_files.items():
        if not check_file_exists(base_path / file, desc):
            all_checks_passed = False
    
    print()
    
    # Check source code files
    print("🔧 Checking Source Code...")
    print("-" * 60)
    src_files = {
        "src/__init__.py": "Package initialization",
        "src/ingestion.py": "Document ingestion module",
        "src/retrieval.py": "Vector store & retrieval module",
        "src/generation.py": "Answer generation module",
        "src/cli.py": "Command-line interface",
        "src/utils.py": "Utility functions",
    }
    
    for file, desc in src_files.items():
        if not check_file_exists(base_path / file, desc):
            all_checks_passed = False
    
    print()
    
    # Check test files
    print("🧪 Checking Test Files...")
    print("-" * 60)
    test_files = {
        "tests/__init__.py": "Tests package initialization",
        "tests/test_ingestion.py": "Ingestion tests",
        "tests/test_retrieval.py": "Retrieval tests",
        "tests/test_generation.py": "Generation tests",
        "tests/test_integration.py": "Integration tests",
    }
    
    for file, desc in test_files.items():
        if not check_file_exists(base_path / file, desc):
            all_checks_passed = False
    
    print()
    
    # Check directories
    print("📁 Checking Data Directories...")
    print("-" * 60)
    directories = {
        "data": "Data directory",
        "src": "Source code directory",
        "tests": "Tests directory",
    }
    
    for dir_name, desc in directories.items():
        if not check_directory_exists(base_path / dir_name, desc):
            all_checks_passed = False
    
    print()
    
    # Check code imports
    print("🔍 Checking Code Imports...")
    print("-" * 60)
    
    try:
        from src.ingestion import DocumentIngester
        print("✅ DocumentIngester imports correctly")
    except Exception as e:
        print(f"❌ DocumentIngester import failed: {e}")
        all_checks_passed = False
    
    try:
        from src.retrieval import VectorStore, Retriever
        print("✅ VectorStore and Retriever import correctly")
    except Exception as e:
        print(f"❌ VectorStore/Retriever import failed: {e}")
        all_checks_passed = False
    
    try:
        from src.generation import AnswerGenerator, RAGPipeline
        print("✅ AnswerGenerator and RAGPipeline import correctly")
    except Exception as e:
        print(f"❌ AnswerGenerator/RAGPipeline import failed: {e}")
        all_checks_passed = False
    
    print()
    
    # Final status
    print("=" * 60)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - Pipeline is ready!")
        print()
        print("Next steps:")
        print("1. Copy .env.example to .env and add your OpenAI API key")
        print("2. Run: python setup.bat (or setup.sh on macOS/Linux)")
        print("3. Place your PDF files in data/documents/")
        print("4. Run: python src/cli.py ingest data/documents")
        print("5. Run: python src/cli.py query 'Your question'")
        print()
        print("To run tests:")
        print("   pytest -v")
        print()
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please review the errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
