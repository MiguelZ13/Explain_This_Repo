from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.db.vector_store import VectorStore

class RAGService:
    def __init__(self):
        self.llm = LLMService()
        self.embedder = EmbeddingService()
        self.store = VectorStore()
        

    def process_query(self, query: str) -> str:
        embedding = self.embedder.generate_embedding(query)
        docs = self.store.search(embedding)
        context = " ".join(docs)
        prompt = f"Context: {context}\nQuestion: {query}"
        response = self.llm.generate_response(prompt)
        return response