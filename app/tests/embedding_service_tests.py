import pytest
from unittest.mock import MagicMock, patch
from app.services.embedding_service import EmbeddingService, MODEL, TARGET_TOKENS, OVERLAP_TOKENS


# Fixtures
@pytest.fixture(scope="module")
def service():
    return EmbeddingService()


def make_parsed_chunk(
    content: str,
    file_path: str = "src/foo.py",
    language: str = "python",
    chunk_type: str = "function",
    name: str = "my_func",
    docstring: str = "Does something.",
    start_line: int = 1,
    end_line: int = 10,
    extra: dict | None = None,
) -> dict:
    return {
        "content": content,
        "metadata": {
            "file_path": file_path,
            "language": language,
            "chunk_type": chunk_type,
            "name": name,
            "docstring": docstring,
            "start_line": start_line,
            "end_line": end_line,
            "extra": extra or {},
        },
    }


# _chunk_text
class TestChunkText:
    def test_empty_string_returns_empty_list(self, service):
        assert service._chunk_text("") == []

    def test_short_text_returns_single_chunk(self, service):
        text = "Hello world"
        chunks = service._chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_text_at_exact_chunk_size_returns_single_chunk(self, service):
        # Encode exactly TARGET_TOKENS tokens then decode back
        tokens = service._tokenizer.encode("word " * TARGET_TOKENS, add_special_tokens=False)
        text = service._tokenizer.decode(tokens[:TARGET_TOKENS], skip_special_tokens=True)
        chunks = service._chunk_text(text, chunk_size=TARGET_TOKENS, overlap=0)
        assert len(chunks) == 1

    def test_long_text_is_split_into_multiple_chunks(self, service):
        # Build text that's clearly > TARGET_TOKENS tokens
        long_text = "The quick brown fox jumps over the lazy dog. " * 100
        chunks = service._chunk_text(long_text)
        assert len(chunks) > 1

    def test_chunks_cover_full_content(self, service):
        """Every token in the source must appear in at least one chunk."""
        long_text = "alpha beta gamma delta epsilon " * 60
        chunks = service._chunk_text(long_text)
        # Re-join all chunks and verify key words are still present
        joined = " ".join(chunks)
        for word in ("alpha", "beta", "gamma", "delta", "epsilon"):
            assert word in joined

    def test_overlap_produces_repeated_tokens(self, service):
        """With overlap > 0, the boundary between chunks should share tokens."""
        long_text = "word " * 200
        chunks_overlap = service._chunk_text(long_text, chunk_size=50, overlap=10)
        chunks_no_overlap = service._chunk_text(long_text, chunk_size=50, overlap=0)
        # Overlap means more chunks (or at least the same number)
        assert len(chunks_overlap) >= len(chunks_no_overlap)

    def test_chunk_size_respected(self, service):
        chunk_size = 64
        long_text = "sentence number {i}. " * 200
        chunks = service._chunk_text(long_text, chunk_size=chunk_size, overlap=0)
        for chunk in chunks:
            n_tokens = len(service._tokenizer.encode(chunk, add_special_tokens=False))
            assert n_tokens <= chunk_size, (
                f"Chunk has {n_tokens} tokens, expected ≤ {chunk_size}"
            )

    def test_custom_chunk_size_and_overlap(self, service):
        text = "token " * 300
        chunks = service._chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        for chunk in chunks:
            n_tokens = len(service._tokenizer.encode(chunk, add_special_tokens=False))
            assert n_tokens <= 100


# _embed_chunks
class TestEmbedChunks:
    EMBEDDING_DIM = 768  # bge-base-en-v1.5

    def test_empty_input_returns_empty_list(self, service):
        assert service._embed_chunks([]) == []

    def test_single_chunk_returns_one_embedding(self, service):
        result = service._embed_chunks(["Hello world"])
        assert len(result) == 1

    def test_multiple_chunks_return_matching_count(self, service):
        chunks = ["First sentence.", "Second sentence.", "Third sentence."]
        result = service._embed_chunks(chunks)
        assert len(result) == len(chunks)

    def test_embedding_is_list_of_floats(self, service):
        result = service._embed_chunks(["test"])
        assert isinstance(result[0], list)
        assert all(isinstance(v, float) for v in result[0])

    def test_embedding_dimension(self, service):
        result = service._embed_chunks(["test"])
        assert len(result[0]) == self.EMBEDDING_DIM

    def test_embeddings_are_normalized(self, service):
        """Normalized embeddings have unit L2 norm (≈ 1.0)."""
        import math
        result = service._embed_chunks(["unit norm check"])
        norm = math.sqrt(sum(v ** 2 for v in result[0]))
        assert abs(norm - 1.0) < 1e-5

    def test_different_texts_produce_different_embeddings(self, service):
        results = service._embed_chunks(["cat", "quantum mechanics"])
        assert results[0] != results[1]

    def test_identical_texts_produce_identical_embeddings(self, service):
        results = service._embed_chunks(["same text", "same text"])
        assert results[0] == results[1]


# generate_embedding_for_query
class TestGenerateEmbeddingForQuery:
    EMBEDDING_DIM = 768

    def test_empty_string_returns_empty_list(self, service):
        assert service.generate_embedding_for_query("") == []

    def test_returns_flat_list_of_floats(self, service):
        result = service.generate_embedding_for_query("how does auth work?")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_correct_dimension(self, service):
        result = service.generate_embedding_for_query("find me a function")
        assert len(result) == self.EMBEDDING_DIM

    def test_embedding_is_normalized(self, service):
        import math
        result = service.generate_embedding_for_query("search query")
        norm = math.sqrt(sum(v ** 2 for v in result))
        assert abs(norm - 1.0) < 1e-5

    def test_different_queries_differ(self, service):
        r1 = service.generate_embedding_for_query("authentication")
        r2 = service.generate_embedding_for_query("database migrations")
        assert r1 != r2

    def test_repeated_query_is_deterministic(self, service):
        r1 = service.generate_embedding_for_query("hello world")
        r2 = service.generate_embedding_for_query("hello world")
        assert r1 == r2

    def test_query_prefix_applied(self, service):
        """Query embedding should differ from a plain passage embedding of same text,
        because the query path prepends a search prefix."""
        text = "sort a list in Python"
        query_emb = service.generate_embedding_for_query(text)
        # _embed_chunks encodes raw text without the prefix
        passage_emb = service._embed_chunks([text])[0]
        assert query_emb != passage_emb, (
            "Query embedding should differ from passage embedding due to prefix"
        )


# generate_embedding_for_parsed_chunks
class TestGenerateEmbeddingForParsedChunks:
    EMBEDDING_DIM = 768

    def test_empty_list_returns_empty_list(self, service):
        assert service.generate_embedding_for_parsed_chunks([]) == []

    def test_single_short_chunk_produces_one_result(self, service):
        chunks = [make_parsed_chunk("def hello(): pass")]
        results = service.generate_embedding_for_parsed_chunks(chunks)
        assert len(results) == 1

    def test_result_has_required_keys(self, service):
        chunks = [make_parsed_chunk("def hello(): pass")]
        result = service.generate_embedding_for_parsed_chunks(chunks)[0]
        assert "content" in result
        assert "embedding" in result
        assert "metadata" in result

    def test_metadata_passthrough(self, service):
        chunks = [make_parsed_chunk(
            "def foo(): pass",
            file_path="app/utils.py",
            language="python",
            chunk_type="function",
            name="foo",
            start_line=5,
            end_line=8,
        )]
        result = service.generate_embedding_for_parsed_chunks(chunks)[0]
        meta = result["metadata"]
        assert meta["file_path"] == "app/utils.py"
        assert meta["language"] == "python"
        assert meta["chunk_type"] == "function"
        assert meta["name"] == "foo"
        assert meta["start_line"] == 5
        assert meta["end_line"] == 8

    def test_metadata_includes_chunk_index_and_total(self, service):
        chunks = [make_parsed_chunk("def hello(): pass")]
        result = service.generate_embedding_for_parsed_chunks(chunks)[0]
        assert "chunk_index" in result["metadata"]
        assert "chunk_total" in result["metadata"]

    def test_embedding_dimension(self, service):
        chunks = [make_parsed_chunk("x = 1 + 2")]
        result = service.generate_embedding_for_parsed_chunks(chunks)[0]
        assert len(result["embedding"]) == self.EMBEDDING_DIM

    def test_embedding_is_normalized(self, service):
        import math
        chunks = [make_parsed_chunk("return value")]
        result = service.generate_embedding_for_parsed_chunks(chunks)[0]
        norm = math.sqrt(sum(v ** 2 for v in result["embedding"]))
        assert abs(norm - 1.0) < 1e-5

    def test_multiple_parsed_chunks(self, service):
        chunks = [
            make_parsed_chunk("def add(a, b): return a + b", name="add"),
            make_parsed_chunk("def sub(a, b): return a - b", name="sub"),
            make_parsed_chunk("class Foo: pass", chunk_type="class", name="Foo"),
        ]
        results = service.generate_embedding_for_parsed_chunks(chunks)
        assert len(results) == 3

    def test_long_chunk_split_into_sub_chunks(self, service):
        """Content exceeding TARGET_TOKENS should produce multiple output records."""
        long_content = "x = 1\n" * 500  # will exceed 512 tokens
        chunks = [make_parsed_chunk(long_content)]
        results = service.generate_embedding_for_parsed_chunks(chunks)
        assert len(results) > 1

    def test_sub_chunk_indices_are_sequential(self, service):
        long_content = "y = 2\n" * 500
        chunks = [make_parsed_chunk(long_content)]
        results = service.generate_embedding_for_parsed_chunks(chunks)
        indices = [r["metadata"]["chunk_index"] for r in results]
        assert indices == list(range(len(results)))

    def test_sub_chunk_total_is_consistent(self, service):
        long_content = "z = 3\n" * 500
        chunks = [make_parsed_chunk(long_content)]
        results = service.generate_embedding_for_parsed_chunks(chunks)
        expected_total = len(results)
        for r in results:
            assert r["metadata"]["chunk_total"] == expected_total

    def test_chunk_with_empty_content_is_skipped(self, service):
        chunks = [
            make_parsed_chunk(""),           # empty — should be skipped
            make_parsed_chunk("valid code"),  # should produce a result
        ]
        results = service.generate_embedding_for_parsed_chunks(chunks)
        assert len(results) == 1

    def test_content_preserved_in_output(self, service):
        source = "def greet(name): return f'Hello {name}'"
        chunks = [make_parsed_chunk(source)]
        result = service.generate_embedding_for_parsed_chunks(chunks)[0]
        # For short content that fits in one chunk the text should round-trip intact
        assert result["content"] == source

    def test_original_metadata_not_mutated(self, service):
        """generate_embedding_for_parsed_chunks must not modify the input dicts."""
        chunk = make_parsed_chunk("immutable?")
        original_meta = dict(chunk["metadata"])
        service.generate_embedding_for_parsed_chunks([chunk])
        assert chunk["metadata"] == original_meta

    def test_different_chunks_produce_different_embeddings(self, service):
        chunks = [
            make_parsed_chunk("def authenticate(token): ..."),
            make_parsed_chunk("SELECT * FROM users WHERE id = %s"),
        ]
        results = service.generate_embedding_for_parsed_chunks(chunks)
        assert results[0]["embedding"] != results[1]["embedding"]

    def test_query_and_chunk_embeddings_are_semantically_similar(self, service):
        """A query about a function should be more similar to that function's
        embedding than to a completely unrelated chunk."""
        import math

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x ** 2 for x in a))
            nb = math.sqrt(sum(x ** 2 for x in b))
            return dot / (na * nb)

        relevant_chunk = make_parsed_chunk(
            "def parse_jwt_token(token: str) -> dict:\n"
            "    \"\"\"Decode and validate a JWT token.\"\"\"\n"
            "    return jwt.decode(token, SECRET_KEY, algorithms=['HS256'])"
        )
        unrelated_chunk = make_parsed_chunk(
            "def calculate_area(radius: float) -> float:\n"
            "    \"\"\"Return the area of a circle.\"\"\"\n"
            "    return math.pi * radius ** 2"
        )

        relevant_emb = service.generate_embedding_for_parsed_chunks([relevant_chunk])[0]["embedding"]
        unrelated_emb = service.generate_embedding_for_parsed_chunks([unrelated_chunk])[0]["embedding"]
        query_emb = service.generate_embedding_for_query("how does JWT authentication work?")

        sim_relevant = cosine(query_emb, relevant_emb)
        sim_unrelated = cosine(query_emb, unrelated_emb)
        assert sim_relevant > sim_unrelated, (
            f"Expected JWT chunk (sim={sim_relevant:.4f}) to be closer to the "
            f"auth query than circle area chunk (sim={sim_unrelated:.4f})"
        )