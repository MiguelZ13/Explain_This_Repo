from app.db.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.services.repo_parser import RepoParser

class IngestionService:
    def __init__(self, store: VectorStore, embedder: EmbeddingService, parser: RepoParser):
        self.store = store
        self.embedder = embedder
        self.parser = parser
    
    def ingest_repo(self, repo_path: str) -> None:
        data = self.parser.parse_repo(repo_path)
        repo_name = data["repo_name"] or "unknown_repo"
        chunks_with_metadata = data["chunks"]

        if not chunks_with_metadata:
            print(f"No content found in repo at {repo_path}")
            return

        texts = [item["content"] for item in chunks_with_metadata]
        
        embeddings = self.embedder._embed_chunks(texts)
        
        repo_id = self.store.get_or_create_repo(repo_name, repo_path)
        for embedding, item in zip(embeddings, chunks_with_metadata):
            self.store.add(
                vector=embedding,
                text=item["content"],
                repo_id=repo_id,
                metadata={k: v for k, v in item.items() if k != "content"}
            )