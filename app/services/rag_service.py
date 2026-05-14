from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.db.vector_store import VectorStore

class RAGService:
    def __init__(self, store: VectorStore):
        self.llm = LLMService()
        self.embedder = EmbeddingService()
        self.store = store
        

    def process_query(self, query: str) -> str:
        embedding = self.embedder.generate_embedding_for_query(query)
        context = self.store.search(embedding)
        response = self.llm.generate_response(query, context)
        return response