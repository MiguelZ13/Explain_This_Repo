from functools import lru_cache
from fastapi import Depends
from app.db.vector_store import VectorStore
from app.services.rag_service import RAGService
from app.services.ingestion_service import IngestionService

# Singleton instance of the vector store
@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore()

def get_rag_service(store=Depends(get_vector_store)) -> RAGService:
    return RAGService(store)

def get_ingestion_service(store=Depends(get_vector_store)) -> IngestionService:
    return IngestionService(store)