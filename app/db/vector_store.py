import os
import json
import sqlite3
import numpy as np
import faiss

EMBEDDING_DIM = 1536 # for OpenAI's text-embedding-3-small
DEFAULT_TOP_K = 5

# Known metadata keys that get their own column in the schema.
# Anything else lands in extra_json.
_KNOWN_KEYS = {"file_path", "language", "chunk_type", "name",
               "docstring", "start_line", "end_line",
               "chunk_index", "chunk_total"}

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


class VectorStore:
    def __init__(self, persist_dir: str = ".vector_store"):
        os.makedirs(persist_dir, exist_ok=True)

        self.persist_dir = persist_dir
        self._index_path = os.path.join(persist_dir, "index.faiss")
        self._db_path    = os.path.join(persist_dir, "chunks.db")

        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row 
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._run_schema()

        # The FAISS row position (0-based) is the chunk `id` in SQLite.
        if os.path.exists(self._index_path):
            self._index = faiss.read_index(self._index_path)
        else:
            self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
    
    def add_repo(
        self,
        repo_name: str,
        repo_path: str
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO repos 
                    (repo_name, repo_path)
                VALUES 
                    (:repo_name, :repo_path)
                """,
                {"repo_name": repo_name, "repo_path": repo_path}
            )

    def add(
        self,
        vector:   list[float],
        text:     str,
        metadata: dict | None = None,
    ) -> None:
        self.add_batch([vector], [text], [metadata or {}])

    def add_batch(
        self,
        vectors:   list[list[float]],
        texts:     list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        if not vectors:
            return

        if metadatas is None:
            metadatas = [{} for _ in vectors]

        if not (len(vectors) == len(texts) == len(metadatas)):
            raise ValueError("vectors, texts, and metadatas must be the same length")

        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        arr /= norms

        rows = [self._metadata_to_row(text, meta)
                for text, meta in zip(texts, metadatas)]

        with self._conn:
            cursor = self._conn.executemany(
                """
                INSERT INTO chunks
                    (text, file_path, language, chunk_type, name, docstring,
                     start_line, end_line, chunk_index, chunk_total, extra_json)
                VALUES
                    (:text, :file_path, :language, :chunk_type, :name, :docstring,
                     :start_line, :end_line, :chunk_index, :chunk_total, :extra_json)
                """,
                rows
            )

            first_rowid = cursor.lastrowid - len(rows) + 1
            expected_faiss_start = self._index.ntotal

            # Validate FAISS / SQLite alignment
            if (first_rowid - 1) != expected_faiss_start:
                raise RuntimeError(
                    f"FAISS/SQLite index misalignment: "
                    f"expected FAISS start {expected_faiss_start}, "
                    f"but first SQLite rowid is {first_rowid} "
                    f"(faiss_pos would be {first_rowid - 1})"
                )

        self._index.add(arr)
        faiss.write_index(self._index, self._index_path)

    def search(
        self,
        query_vector: list[float],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        if self._index.ntotal == 0:
            return []

        top_k = min(top_k, self._index.ntotal)
        arr   = self._to_unit_array(query_vector)
        scores, indices = self._index.search(arr, top_k)

        results = []
        for score, faiss_idx in zip(scores[0], indices[0]):
            if faiss_idx == -1:
                continue

            # FAISS position → SQLite id (0-based → 1-based)
            sqlite_id = int(faiss_idx) + 1
            row = self._conn.execute(
                "SELECT * FROM chunks WHERE id = ?", (sqlite_id,)
            ).fetchone()

            if row is None:
                continue

            results.append({
                "text":     row["text"],
                "score":    float(score),
                "metadata": self._row_to_metadata(row),
            })

        return results

    def get_by_repo(self, repo_name: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE file_path LIKE ?",
            (f"{repo_name}/%",)
        ).fetchall()
        return [{"text": r["text"], "metadata": self._row_to_metadata(r)} for r in rows]

    @property
    def size(self) -> int:
        return self._index.ntotal

    def _run_schema(self) -> None:
        with open(_SCHEMA_PATH, "r") as f:
            sql = f.read()
        with self._conn:
            self._conn.executescript(sql)

    @staticmethod
    def _metadata_to_row(text: str, metadata: dict) -> dict:
        extra = {k: v for k, v in metadata.items() if k not in _KNOWN_KEYS}
        return {
            "text":        text,
            "file_path":   metadata.get("file_path"),
            "language":    metadata.get("language"),
            "chunk_type":  metadata.get("chunk_type"),
            "name":        metadata.get("name"),
            "docstring":   metadata.get("docstring"),
            "start_line":  metadata.get("start_line"),
            "end_line":    metadata.get("end_line"),
            "chunk_index": metadata.get("chunk_index"),
            "chunk_total": metadata.get("chunk_total"),
            "extra_json":  json.dumps(extra),
        }

    @staticmethod
    def _row_to_metadata(row: sqlite3.Row) -> dict:
        extra = json.loads(row["extra_json"] or "{}")
        meta  = {
            "file_path":   row["file_path"],
            "language":    row["language"],
            "chunk_type":  row["chunk_type"],
            "name":        row["name"],
            "docstring":   row["docstring"],
            "start_line":  row["start_line"],
            "end_line":    row["end_line"],
            "chunk_index": row["chunk_index"],
            "chunk_total": row["chunk_total"],
        }
        meta = {k: v for k, v in meta.items() if v is not None}
        meta.update(extra)
        return meta

    @staticmethod
    def _to_unit_array(vector: list[float]) -> np.ndarray:
        arr  = np.array(vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        return arr

    def close(self) -> None:
        self._conn.close()