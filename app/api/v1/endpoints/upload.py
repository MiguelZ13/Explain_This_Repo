from fastapi import APIRouter, Depends
from pydantic import BaseModel, HttpUrl
from app.api.deps import get_ingestion_service


router = APIRouter()

class RepoURLRequest(BaseModel):
    url: HttpUrl

@router.post("/")
async def upload_repo(request: RepoURLRequest, ingestor = Depends(get_ingestion_service)):
    ingestor.ingest_repo(request.url)
    return {"status": "ok", "message": f"Repository at {request.url} has been ingested successfully."}
