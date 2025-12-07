from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


class AnswerGenerator:
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
    
    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        context_text = self._format_context(context)
        
        system_prompt = SystemMessage(content="""You are a helpful assistant that answers questions based on provided documents.
- Answer the user's question using only information from the provided context.
- If the context doesn't contain information to answer the question, say so clearly.
- Cite the source document when using specific information.
- Be concise and clear.""")
        
        user_prompt = HumanMessage(content=f"""Context:
{context_text}

Question: {query}

Answer:""")
        
        response = self.llm([system_prompt, user_prompt])
        return response.content.strip()
    
    @staticmethod
    def _format_context(context: List[Dict[str, Any]]) -> str:
        formatted = []
        for i, doc in enumerate(context, 1):
            formatted.append(f"Document {i} (from {doc['source']}):\n{doc['content']}\n")
        return "\n".join(formatted)


class RAGPipeline:
    def __init__(self, retriever, answer_generator: AnswerGenerator):
        self.retriever = retriever
        self.generator = answer_generator
    
    def answer(self, query: str, k: int = 4) -> Dict[str, Any]:
        retrieved = self.retriever.retrieve(query, k=k)
        
        answer = self.generator.generate(query, retrieved)
        
        return {
            'query': query,
            'retrieved_documents': retrieved,
            'answer': answer,
            'num_sources': len(retrieved)
        }
