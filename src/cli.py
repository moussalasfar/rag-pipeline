import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.ingestion import DocumentIngester
from src.retrieval import VectorStore, Retriever
from src.generation import AnswerGenerator, RAGPipeline


def ingest_command(data_dir: str):
    if not Path(data_dir).exists():
        print(f"❌ Directory not found: {data_dir}")
        return
    
    ingester = DocumentIngester()
    chunks = ingester.ingest_directory(data_dir)
    
    if not chunks:
        print("❌ No documents found to ingest")
        return
    
    vector_store = VectorStore()
    vector_store.add_documents(chunks)
    vector_store.save()
    
    print(f"✓ Successfully ingested {len(chunks)} chunks from {len(set(c.metadata['source'] for c in chunks))} documents")


def query_command(query: str, k: int = 4):
    vector_store = VectorStore()
    try:
        vector_store.load()
    except FileNotFoundError:
        print("❌ No index found. Run 'python src/cli.py ingest data/documents' first.")
        return
    
    retriever = Retriever(vector_store)
    generator = AnswerGenerator()
    rag = RAGPipeline(retriever, generator)
    
    print(f"\n🔍 Query: {query}\n")
    result = rag.answer(query, k=k)
    
    print("📄 Retrieved Documents:")
    for i, doc in enumerate(result['retrieved_documents'], 1):
        print(f"\n  [{i}] {doc['source']} (score: {doc['score']:.3f})")
        print(f"      {doc['content'][:200]}...")
    
    print(f"\n💡 Answer:")
    print(f"   {result['answer']}\n")


def main():
    if len(sys.argv) < 2:
        print("""
Usage: python src/cli.py [command] [args]

Commands:
  ingest <directory>  - Ingest all PDFs from a directory
  query <query>       - Query the knowledge base
  
Examples:
  python src/cli.py ingest data/documents
  python src/cli.py query "What is the main topic?"
        """)
        return
    
    command = sys.argv[1]
    
    if command == "ingest":
        if len(sys.argv) < 3:
            print("❌ Please specify a directory: python src/cli.py ingest <directory>")
            return
        ingest_command(sys.argv[2])
    
    elif command == "query":
        if len(sys.argv) < 3:
            print("❌ Please specify a query: python src/cli.py query '<query>'")
            return
        query_text = " ".join(sys.argv[2:])
        k = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 4
        query_command(query_text, k=k)
    
    else:
        print(f"❌ Unknown command: {command}")


if __name__ == "__main__":
    main()
