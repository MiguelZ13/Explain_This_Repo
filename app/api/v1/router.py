from fastapi import APIRouter
from app.api.v1.endpoints import upload, query

api_router = APIRouter()
api_router.include_router(upload.router, prefix="/upload", tags=["upload"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
