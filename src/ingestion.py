"""Document ingestion pipeline: PDF parsing, chunking, and metadata extraction."""

from pathlib import Path
from typing import List, Dict, Any
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentIngester:
    """Handles document loading and chunking."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize ingester with chunking parameters.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def ingest_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and chunk a PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of chunks with metadata
        """
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        chunks = self.splitter.split_documents(docs)
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['source'] = str(file_path)
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        return chunks
    
    def ingest_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        Load and chunk all PDF files in a directory.
        
        Args:
            directory: Path to directory containing PDFs
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        path = Path(directory)
        
        for pdf_file in path.glob("*.pdf"):
            print(f"Ingesting {pdf_file.name}...")
            chunks = self.ingest_pdf(str(pdf_file))
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)} chunks extracted")
        
        return all_chunks
    
    def ingest_text(self, text: str, source: str = "text_input") -> List[Dict[str, Any]]:
        """
        Chunk plain text.
        
        Args:
            text: Text content to chunk
            source: Source identifier for metadata
            
        Returns:
            List of chunks with metadata
        """
        from langchain.schema import Document
        
        doc = Document(page_content=text, metadata={"source": source})
        chunks = self.splitter.split_documents([doc])
        
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        return chunks
