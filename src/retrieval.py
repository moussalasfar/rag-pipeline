"""Vector store and retrieval pipeline."""

import os
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document


class VectorStore:
    """FAISS-based vector store for semantic search."""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small", index_path: str = None):
        """
        Initialize vector store with embeddings.
        
        Args:
            embedding_model: OpenAI embedding model name
            index_path: Path to save/load FAISS index
        """
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.index = None
        self.documents = []
        self.index_path = index_path or "data/index/faiss_index"
        self.embeddings_path = index_path.replace(".faiss", "_embeddings.npy") if index_path else "data/index/embeddings.npy"
    
    def add_documents(self, docs: List[Document]) -> None:
        """
        Add documents to vector store.
        
        Args:
            docs: List of Document objects
        """
        print(f"Generating embeddings for {len(docs)} chunks...")
        texts = [doc.page_content for doc in docs]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        
        # Store documents and embeddings
        self.documents = docs
        self.embeddings_array = embeddings_array
        
        print(f"✓ Index created with {len(docs)} documents")
    
    def save(self) -> None:
        """Save index and embeddings to disk."""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, self.index_path)
        np.save(self.embeddings_path, self.embeddings_array)
        
        # Save document metadata as text
        metadata_path = self.index_path.replace(".faiss", "_metadata.txt")
        with open(metadata_path, 'w') as f:
            for i, doc in enumerate(self.documents):
                source = doc.metadata.get('source', 'unknown')
                chunk_id = doc.metadata.get('chunk_id', i)
                f.write(f"{i}|{source}|{chunk_id}\n")
        
        print(f"✓ Index saved to {self.index_path}")
    
    def load(self) -> None:
        """Load index and embeddings from disk."""
        if not Path(self.index_path).exists():
            raise FileNotFoundError(f"Index not found at {self.index_path}")
        
        self.index = faiss.read_index(self.index_path)
        self.embeddings_array = np.load(self.embeddings_path)
        
        # Load document metadata
        metadata_path = self.index_path.replace(".faiss", "_metadata.txt")
        self.documents = []
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                for line in f:
                    idx, source, chunk_id = line.strip().split('|')
                    self.documents.append(Document(
                        page_content="",
                        metadata={'source': source, 'chunk_id': int(chunk_id)}
                    ))
        
        print(f"✓ Index loaded from {self.index_path}")
    
    def search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Call add_documents() or load() first.")
        
        query_embedding = self.embeddings.embed_query(query)
        query_array = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_array, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx]
                score = float(1 / (1 + distance))  # Convert distance to similarity
                results.append((doc, score))
        
        return results


class Retriever:
    """High-level retriever interface."""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize retriever.
        
        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
    
    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of result dicts with content and metadata
        """
        results = self.vector_store.search(query, k=k)
        
        return [
            {
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'unknown'),
                'score': score
            }
            for doc, score in results
        ]
