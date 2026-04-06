from fastapi import APIRouter
from app.services.rag_service import RAGService

router = APIRouter()
rag_service = RAGService()

@router.post("/")
async def query_repo(query: str):
    response = rag_service.query(query)
    return {"response": response}
