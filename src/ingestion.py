from pathlib import Path
from typing import List, Dict, Any
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentIngester:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def ingest_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        chunks = self.splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata['source'] = str(file_path)
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        return chunks
    
    def ingest_directory(self, directory: str) -> List[Dict[str, Any]]:
        all_chunks = []
        path = Path(directory)
        
        for pdf_file in path.glob("*.pdf"):
            print(f"Ingesting {pdf_file.name}...")
            chunks = self.ingest_pdf(str(pdf_file))
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)} chunks extracted")
        
        return all_chunks
    
    def ingest_text(self, text: str, source: str = "text_input") -> List[Dict[str, Any]]:
        from langchain.schema import Document
        
        doc = Document(page_content=text, metadata={"source": source})
        chunks = self.splitter.split_documents([doc])
        
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        return chunks
