from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-base-en-v1.5"
TARGET_TOKENS = 512
OVERLAP_TOKENS = 64 

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(MODEL)
        self._tokenizer = self.model.tokenizer
    
    def _chunk_text(self, text: str, chunk_size: int = TARGET_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
        # Tokenizes the text and splits it into chunks of approximately chunk_size tokens with overlap
        tokens = self._tokenizer.encode(text, add_special_tokens=False)

        if not tokens:
            return []

        if len(tokens) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self._tokenizer.decode(chunk_tokens, skip_special_tokens=True))
            if end == len(tokens):
                break
            start += chunk_size - overlap

        return chunks
    
    def _embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []

        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        return embeddings.tolist()

    # Designed for simple text inputs where we just want to generate an embedding for the entire text (used for queries).
    def generate_embedding_for_query(self, text: str) -> list[float]:
        if not text:
            return []
        
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        embedding = self.model.encode([prefixed], normalize_embeddings=True)
        return embedding[0].tolist()
    
    # This method is designed to take the parsed chunks from the RepoParser, which include both content and metadata, 
    # and generate embeddings for each chunk while preserving the associated metadata. 
    # The resulting list of dictionaries includes: original content, its embedding vector, and metadata that 
    # also indicates the chunk's position within the original content.
    def generate_embedding_for_parsed_chunks(self, chunks: list[dict]) -> list[dict]:
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
