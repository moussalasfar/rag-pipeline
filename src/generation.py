"""LLM-powered answer generation with retrieved context."""

from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


class AnswerGenerator:
    """Generates answers using LLM with retrieved context."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """
        Initialize answer generator.
        
        Args:
            model: OpenAI model name
            temperature: Sampling temperature (0-1)
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
    
    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate an answer based on query and retrieved context.
        
        Args:
            query: User query
            context: List of retrieved documents with 'content' and 'source' keys
            
        Returns:
            Generated answer string
        """
        # Format context
        context_text = self._format_context(context)
        
        # Create prompt
        system_prompt = SystemMessage(content="""You are a helpful assistant that answers questions based on provided documents.
- Answer the user's question using only information from the provided context.
- If the context doesn't contain information to answer the question, say so clearly.
- Cite the source document when using specific information.
- Be concise and clear.""")
        
        user_prompt = HumanMessage(content=f"""Context:
{context_text}

Question: {query}

Answer:""")
        
        # Generate answer
        response = self.llm([system_prompt, user_prompt])
        return response.content.strip()
    
    @staticmethod
    def _format_context(context: List[Dict[str, Any]]) -> str:
        """Format retrieved documents for LLM context."""
        formatted = []
        for i, doc in enumerate(context, 1):
            formatted.append(f"Document {i} (from {doc['source']}):\n{doc['content']}\n")
        return "\n".join(formatted)


class RAGPipeline:
    """End-to-end RAG pipeline."""
    
    def __init__(self, retriever, answer_generator: AnswerGenerator):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Retriever instance
            answer_generator: AnswerGenerator instance
        """
        self.retriever = retriever
        self.generator = answer_generator
    
    def answer(self, query: str, k: int = 4) -> Dict[str, Any]:
        """
        Answer a query using RAG pipeline.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Dict with query, retrieved docs, and answer
        """
        # Retrieve context
        retrieved = self.retriever.retrieve(query, k=k)
        
        # Generate answer
        answer = self.generator.generate(query, retrieved)
        
        return {
            'query': query,
            'retrieved_documents': retrieved,
            'answer': answer,
            'num_sources': len(retrieved)
        }
