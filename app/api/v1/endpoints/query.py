from fastapi import APIRouter, Depends
from app.api.deps import get_rag_service


router = APIRouter()
@router.post("/")
async def query_repo(query: str, rag = Depends(get_rag_service)):
    response = rag.query(query)
    return {"response": response}
