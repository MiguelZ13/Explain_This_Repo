from openai import OpenAI
import tiktoken

MODEL = "text-embedding-3-small"
MAX_TOKENS = 8191  # Max tokens for text-embedding-3-small
TARGET_TOKENS = 512
OVERLAP_TOKENS = 64 

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI()
        self._tokenizer = tiktoken.encoding_for_model(MODEL)
    
    def _chunk_code(self, code: str, chunk_size: int = TARGET_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
        tokens = self._tokenizer.encode(code)

        if not tokens:
            return []

        if len(tokens) <= chunk_size:
            return [code]

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

    def generate_embedding(self, text: str) -> list[list[float]]:
        chunks = self._chunk_code(text)
        if not chunks:
            return []

        embeddings = self._embed_chunks(chunks)
        return embeddings
    
    def generate_embedding_for_parsed_chunks(self, chunks: list[dict]) -> list[dict]:
        results: list[dict] = []

        for chunk in chunks:
            content  = chunk["content"]
            metadata = chunk["metadata"]

            text_chunks = self._chunk_code(content)
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
