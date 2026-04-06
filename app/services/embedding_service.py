from openai import OpenAI

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI()
    
    def _chunk_code(self, code: str, chunk_size: int = 100) -> list:
        return [code[i:i+chunk_size] for i in range(0, len(code), chunk_size)]

    def generate_embedding(self, text: str):
        chunks = self._chunk_code(text)
        embeddings = [
            self.client.embeddings.create(
                input=chunk, 
                model="text-embedding-3-small"
            ).data[0].embedding()
            for chunk in chunks
        ]
        return embeddings
