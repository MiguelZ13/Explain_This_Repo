import tempfile
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional
from tree_sitter import Language, Parser, Node


def _load_languages() -> dict[str, Language]:

    langs: dict[str, Language] = {}

    grammar_packages = {
        "python":     ("tree_sitter_python",     "language"),
        "javascript": ("tree_sitter_javascript", "language"),
        "typescript": ("tree_sitter_typescript", "language_typescript"),
        "tsx":        ("tree_sitter_typescript", "language_tsx"),
        "java":       ("tree_sitter_java",       "language"),
        "go":         ("tree_sitter_go",         "language"),
        "rust":       ("tree_sitter_rust",       "language"),
        "cpp":        ("tree_sitter_cpp",        "language"),
        "c":          ("tree_sitter_c",          "language"),
        "csharp":     ("tree_sitter_c_sharp",    "language"),
        "ruby":       ("tree_sitter_ruby",       "language"),
        "php":        ("tree_sitter_php",        "language_php"),
        "swift":      ("tree_sitter_swift",      "language"),
        "kotlin":     ("tree_sitter_kotlin",     "language"),
        "bash":       ("tree_sitter_bash",       "language"),
        "lua":        ("tree_sitter_lua",        "language"),
        "scala":      ("tree_sitter_scala",      "language"),
        "haskell":    ("tree_sitter_haskell",    "language"),
        "elixir":     ("tree_sitter_elixir",     "language"),
        "r":          ("tree_sitter_r",          "language"),
    }

    for lang_name, (module_name, attr) in grammar_packages.items():
        try:
            import importlib
            mod = importlib.import_module(module_name)
            langs[lang_name] = Language(getattr(mod, attr)())
        except Exception:
            print(f"Warning: Failed to load grammar for {lang_name} from {module_name}.{attr}")
            pass

    return langs


LANGUAGES: dict[str, Language] = _load_languages()


EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py":    "python",
    ".js":    "javascript",
    ".mjs":   "javascript",
    ".cjs":   "javascript",
    ".jsx":   "javascript",
    ".ts":    "typescript",
    ".tsx":   "tsx",
    ".java":  "java",
    ".go":    "go",
    ".rs":    "rust",
    ".cpp":   "cpp",
    ".cc":    "cpp",
    ".cxx":   "cpp",
    ".c":     "c",
    ".h":     "c",
    ".hpp":   "cpp",
    ".cs":    "csharp",
    ".rb":    "ruby",
    ".php":   "php",
    ".swift": "swift",
    ".kt":    "kotlin",
    ".kts":   "kotlin",
    ".sh":    "bash",
    ".bash":  "bash",
    ".lua":   "lua",
    ".scala": "scala",
    ".hs":    "haskell",
    ".ex":    "elixir",
    ".exs":   "elixir",
    ".r":     "r",
    ".R":     "r",

    ".md":    "markdown",
    ".txt":   "text",
    ".json":  "json",
    ".yaml":  "yaml",
    ".yml":   "yaml",
    ".toml":  "toml",
    ".html":  "html",
    ".css":   "css",
}

SKIP_DIRS: set[str] = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".idea", ".vscode", ".mypy_cache", "vendor",
}

SKIP_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".lock", ".pyc", ".pyo", ".exe", ".bin",
    ".so", ".dylib", ".dll", ".wasm", ".pdf", ".zip", 
    ".tar", ".gz", ".7z", ".mp4", ".mp3", ".avi", ".mov", ".wmv"
}


_QUERIES: dict[str, list[tuple[str, str]]] = {

    "python": [
        ("function", """
            (function_definition
                name: (identifier) @name
                body: (block
                    (expression_statement
                        (string) @docstring)?)
            ) @definition
        """),
        ("class", """
            (class_definition
                name: (identifier) @name
                body: (block
                    (expression_statement
                        (string) @docstring)?)
            ) @definition
        """),
    ],

    "javascript": [
        ("function", """
            [
              (function_declaration   name: (identifier) @name) @definition
              (method_definition      name: (property_identifier) @name) @definition
              (arrow_function) @definition
            ]
        """),
        ("class", """
            (class_declaration name: (identifier) @name) @definition
        """),
    ],

    "typescript": [
        ("function", """
            [
              (function_declaration   name: (identifier) @name) @definition
              (method_definition      name: (property_identifier) @name) @definition
              (arrow_function) @definition
            ]
        """),
        ("class", """
            (class_declaration name: (identifier) @name) @definition
        """),
        ("interface", """
            (interface_declaration name: (type_identifier) @name) @definition
        """),
    ],

    "tsx": [
        ("function", """
            [
              (function_declaration   name: (identifier) @name) @definition
              (method_definition      name: (property_identifier) @name) @definition
            ]
        """),
        ("class", """
            (class_declaration name: (identifier) @name) @definition
        """),
    ],

    "java": [
        ("function", """
            (method_declaration name: (identifier) @name) @definition
        """),
        ("class", """
            (class_declaration name: (identifier) @name) @definition
        """),
    ],

    "go": [
        ("function", """
            [
              (function_declaration  name: (identifier) @name) @definition
              (method_declaration    name: (field_identifier) @name) @definition
            ]
        """),
        ("struct", """
            (type_declaration
                (type_spec name: (type_identifier) @name
                    type: (struct_type))
            ) @definition
        """),
    ],

    "rust": [
        ("function", """
            (function_item name: (identifier) @name) @definition
        """),
        ("struct", """
            (struct_item name: (type_identifier) @name) @definition
        """),
        ("impl", """
            (impl_item
                type: (type_identifier) @name
            ) @definition
        """),
    ],

    "cpp": [
        ("function", """
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @name)
            ) @definition
        """),
        ("class", """
            (class_specifier name: (type_identifier) @name) @definition
        """),
    ],

    "c": [
        ("function", """
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @name)
            ) @definition
        """),
    ],

    "csharp": [
        ("function", """
            (method_declaration name: (identifier) @name) @definition
        """),
        ("class", """
            (class_declaration name: (identifier) @name) @definition
        """),
    ],

    "ruby": [
        ("function", """
            (method name: (identifier) @name) @definition
        """),
        ("class", """
            (class name: (constant) @name) @definition
        """),
    ],

    "kotlin": [
        ("function", """
            (function_declaration (simple_identifier) @name) @definition
        """),
        ("class", """
            (class_declaration (type_identifier) @name) @definition
        """),
    ],

    "swift": [
        ("function", """
            (function_declaration name: (simple_identifier) @name) @definition
        """),
        ("class", """
            (class_declaration name: (type_identifier) @name) @definition
        """),
    ],

    "scala": [
        ("function", """
            (function_definition name: (identifier) @name) @definition
        """),
        ("class", """
            (class_definition name: (identifier) @name) @definition
        """),
    ],

    "elixir": [
        ("function", """
            (call
                target: (identifier) @_def (#match? @_def "^def(p)?$")
                (arguments (call target: (identifier) @name))
            ) @definition
        """),
    ],
}


@dataclass
class ParsedChunk:
    content: str
    file_path: str
    language: str
    chunk_type: str
    name: Optional[str] = None
    docstring: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    extra: dict = field(default_factory=dict)


class RepoParser:

    def __init__(self):
        self._parsers: dict[str, Parser] = {}


    def parse_repo(self, repo_url: str) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_name = os.path.basename(repo_url).replace(".git", "")
            repo_path = os.path.join(tmpdir, repo_name)
            subprocess.run(["git", "clone", repo_url, repo_path], check=True)

            chunks: list[ParsedChunk] = []

            for root, dirs, files in os.walk(repo_path):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in SKIP_EXTENSIONS:
                        continue

                    abs_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(abs_path, repo_path)
                    language = EXTENSION_TO_LANGUAGE.get(ext, "unknown")

                    try:
                        source = abs_path
                        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                            source = f.read()
                    except Exception:
                        print(f"Warning: Failed to read {rel_path}")
                        continue

                    if not source.strip():
                        continue

                    file_chunks = self._parse_file(source, rel_path, language)
                    chunks.extend(file_chunks)

            return {
                "repo_name": repo_name,
                "chunks": [self._to_dict(c) for c in chunks],
            }

    def _parse_file(self, source: str, file_path: str, language: str) -> list[ParsedChunk]:
        if language in LANGUAGES and language in _QUERIES:
            return self._parse_with_tree_sitter(source, file_path, language)

        return [ParsedChunk(
            content=source,
            file_path=file_path,
            language=language,
            chunk_type="module",
        )]

    def _get_parser(self, language: str) -> Parser:
        if language not in self._parsers:
            parser = Parser(LANGUAGES[language])
            self._parsers[language] = parser
        return self._parsers[language]

    def _parse_with_tree_sitter(
        self, source: str, file_path: str, language: str
    ) -> list[ParsedChunk]:

        parser = self._get_parser(language)
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)

        queries = _QUERIES[language]
        chunks: list[ParsedChunk] = []
        covered_ranges: list[tuple[int, int]] = [] 

        for chunk_type, query_str in queries:
            try:
                query = LANGUAGES[language].query(query_str)
            except Exception:
                print(f"Warning: Failed to compile query for {language} - {chunk_type}")
                continue

            matches = query.matches(tree.root_node)

            for _pattern_index, capture_dict in matches:
                def_nodes: list[Node] = capture_dict.get("definition", [])
                name_nodes: list[Node] = capture_dict.get("name", [])
                doc_nodes: list[Node]  = capture_dict.get("docstring", [])

                if isinstance(def_nodes, Node):
                    def_nodes = [def_nodes]
                if isinstance(name_nodes, Node):
                    name_nodes = [name_nodes]
                if isinstance(doc_nodes, Node):
                    doc_nodes = [doc_nodes]

                for def_node in def_nodes:
                    start_byte = def_node.start_byte
                    end_byte   = def_node.end_byte

                    if any(s <= start_byte and end_byte <= e for s, e in covered_ranges):
                        continue
                    covered_ranges.append((start_byte, end_byte))

                    content   = source_bytes[start_byte:end_byte].decode("utf-8", errors="ignore")
                    name      = self._node_text(name_nodes[0], source_bytes) if name_nodes else None
                    docstring = self._extract_docstring(doc_nodes, source_bytes, language)

                    chunks.append(ParsedChunk(
                        content=content,
                        file_path=file_path,
                        language=language,
                        chunk_type=chunk_type,
                        name=name,
                        docstring=docstring,
                        start_line=def_node.start_point[0] + 1,
                        end_line=def_node.end_point[0] + 1,
                    ))

        if not chunks:
            chunks.append(ParsedChunk(
                content=source,
                file_path=file_path,
                language=language,
                chunk_type="module",
            ))

        return chunks


    @staticmethod
    def _node_text(node: Node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore").strip()

    @staticmethod
    def _extract_docstring(
        doc_nodes: list[Node], source_bytes: bytes, language: str
    ) -> Optional[str]:
        if not doc_nodes:
            return None
        raw = source_bytes[doc_nodes[0].start_byte:doc_nodes[0].end_byte].decode("utf-8", errors="ignore").strip()
        if language in ("python", "javascript", "typescript", "tsx"):
            raw = raw.strip("\"'")
            if raw.startswith('"""') or raw.startswith("'''"):
                raw = raw[3:]
            if raw.endswith('"""') or raw.endswith("'''"):
                raw = raw[:-3]
        elif raw.startswith(("//", "#")):
            raw = "\n".join(
                line.lstrip("#/ \t") for line in raw.splitlines()
            )
        return raw.strip() or None

    @staticmethod
    def _to_dict(chunk: ParsedChunk) -> dict:
        return {
            "content": chunk.content,
            "metadata": {
                "file_path":  chunk.file_path,
                "language":   chunk.language,
                "chunk_type": chunk.chunk_type,
                "name":       chunk.name,
                "docstring":  chunk.docstring,
                "start_line": chunk.start_line,
                "end_line":   chunk.end_line,
                **chunk.extra,
            },
        }