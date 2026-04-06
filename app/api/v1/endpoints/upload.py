from fastapi import APIRouter
from pydantic import BaseModel, HttpUrl
from app.services.repo_parser import RepoParser

router = APIRouter()
parser = RepoParser()

class RepoURLRequest(BaseModel):
    url: HttpUrl

@router.post("/")
async def upload_repo(request: RepoURLRequest):
    repo_info = parser.parse_repo(str(request.url))
    return {"repo_info": repo_info}
