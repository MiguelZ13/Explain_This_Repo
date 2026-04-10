from app.services.ingestion_service import IngestionService
from fastapi import APIRouter
from pydantic import BaseModel, HttpUrl
from app.services.repo_parser import RepoParser
from app.services.embedding_service import EmbeddingService
from app.db.vector_store import VectorStore


router = APIRouter()
parser = RepoParser()
embedder = EmbeddingService()
vector_store = VectorStore()
ingestor = IngestionService(vector_store, embedder, parser)

class RepoURLRequest(BaseModel):
    url: HttpUrl

@router.post("/")
async def upload_repo(request: RepoURLRequest):
    ingestor.ingest_repo(request.url)
    return {"status": "ok", "message": f"Repository at {request.url} has been ingested successfully."}
