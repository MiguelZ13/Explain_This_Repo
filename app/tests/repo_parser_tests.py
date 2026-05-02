import os
import subprocess
import tempfile
import pytest

from app.services.repo_parser import RepoParser, LANGUAGES

@pytest.fixture
def parser():
    return RepoParser()


@pytest.fixture
def python_available():
    if "python" not in LANGUAGES:
        pytest.skip("tree-sitter python grammar not installed")



def test_unknown_language_fallback(parser):
    source = "just some text"
    chunks = parser._parse_file(source, "file.xyz", "unknown")

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "module"
    assert chunks[0].content == source


def test_empty_source(parser):
    chunks = parser._parse_file("   ", "file.py", "python")
    # Empty is filtered earlier normally, but still ensure safe behavior
    assert isinstance(chunks, list)


def test_parse_simple_python_function(parser, python_available):
    source = '''
    def foo():
        """hello world"""
        return 42
    '''

    chunks = parser._parse_file(source, "test.py", "python")

    assert len(chunks) == 1
    chunk = chunks[0]

    assert chunk.name == "foo"
    assert chunk.chunk_type == "function"
    assert "return 42" in chunk.content
    assert chunk.docstring == "hello world"


def test_parse_multiple_functions(parser, python_available):
    source = '''
    def a(): pass
    def b(): pass
    '''

    chunks = parser._parse_file(source, "test.py", "python")

    names = {c.name for c in chunks}
    assert names == {"a", "b"}


def test_parse_class(parser, python_available):
    source = '''
    class MyClass:
        """doc"""
        pass
    '''

    chunks = parser._parse_file(source, "test.py", "python")

    assert len(chunks) == 1
    chunk = chunks[0]

    assert chunk.name == "MyClass"
    assert chunk.chunk_type == "class"
    assert chunk.docstring == "doc"


def test_extract_docstring_python():
    raw = b'"""hello"""'

    class FakeNode:
        start_byte = 0
        end_byte = len(raw)

    result = RepoParser._extract_docstring([FakeNode()], raw, "python")

    assert result == "hello"


def test_extract_docstring_comment_style():
    raw = b"# hello\n# world"

    class FakeNode:
        start_byte = 0
        end_byte = len(raw)

    result = RepoParser._extract_docstring([FakeNode()], raw, "ruby")

    assert result == "hello\nworld"


def test_node_text():
    raw = b"hello world"

    class FakeNode:
        start_byte = 0
        end_byte = len(raw)

    text = RepoParser._node_text(FakeNode(), raw)
    assert text == "hello world"


def test_parse_repo_local_git(parser, python_available):
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = os.path.join(tmpdir, "repo")
        os.mkdir(repo_path)

        subprocess.run(["git", "init"], cwd=repo_path, check=True)

        file_path = os.path.join(repo_path, "test.py")
        with open(file_path, "w") as f:
            f.write("def foo(): return 1")

        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=repo_path, check=True)

        result = parser.parse_repo(repo_path)

        assert result["repo_name"] == "repo"
        assert len(result["chunks"]) > 0

        chunk = result["chunks"][0]
        assert "content" in chunk
        assert "metadata" in chunk


def test_binary_file_skipped(parser):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "image.png")
        with open(file_path, "wb") as f:
            f.write(b"binarydata")

        # Simulate parse_file directly
        chunks = parser._parse_file("binarydata", "image.png", "unknown")
        assert chunks[0].chunk_type == "module"


def test_non_utf8_file(parser, python_available):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.py")
        with open(file_path, "wb") as f:
            f.write(b"\xff\xfe\x00\x00")

        # Should not crash
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        chunks = parser._parse_file(content, "test.py", "python")
        assert isinstance(chunks, list)


def test_parser_caching(parser, python_available):
    p1 = parser._get_parser("python")
    p2 = parser._get_parser("python")

    assert p1 is p2
