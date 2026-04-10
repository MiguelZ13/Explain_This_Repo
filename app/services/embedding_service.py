from openai import OpenAI
import tiktoken
from app.db.vector_store import VectorStore

MODEL = "text-embedding-3-small"
MAX_TOKENS = 8191  # Max tokens for text-embedding-3-small
TARGET_TOKENS = 512
OVERLAP_TOKENS = 64 

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI()
        self._tokenizer = tiktoken.encoding_for_model(MODEL)
    
    def _chunk_text(self, text: str, chunk_size: int = TARGET_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
        # Tokenizes the text and splits it into chunks of approximately chunk_size tokens with overlap
        tokens = self._tokenizer.encode(text)

        if not tokens:
            return []

        if len(tokens) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self._tokenizer.decode(chunk_tokens))
            if end == len(tokens):
                break
            start += chunk_size - overlap

        return chunks
    
    def _embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []

        response = self.client.embeddings.create(
            input=chunks,
            model=MODEL,
        )

        return [item.embedding for item in response.data]

    # Designed for simple text inputs where we just want to generate an embedding for the entire text (used for queries).
    def generate_embedding_for_query(self, text: str) -> list[list[float]]:
        if not text:
            return []
        
        response = self.client.embeddings.create(
            input=[text],
            model=MODEL,
        )
        return response.data[0].embedding
    
    # This method is designed to take the parsed chunks from the RepoParser, which include both content and metadata, 
    # and generate embeddings for each chunk while preserving the associated metadata. 
    # The resulting list of dictionaries includes: original content, its embedding vector, and metadata that 
    # also indicates the chunk's position within the original content.
    def generate_embedding_for_parsed_chunks(self, chunks: list[dict], store: VectorStore) -> list[dict]:
        results: list[dict] = []

        for chunk in chunks:
            content  = chunk["content"]
            metadata = chunk["metadata"]

            text_chunks = self._chunk_text(content)
            if not text_chunks:
                continue

            embeddings = self._embed_chunks(text_chunks)

            for i, (chunk_text, vector) in enumerate(zip(text_chunks, embeddings)):
                results.append({
                    "content":   chunk_text,
                    "embedding": vector,
                    "metadata":  {
                        **metadata,
                        "chunk_index":  i,
                        "chunk_total":  len(text_chunks),
                    },
                })

        return results
