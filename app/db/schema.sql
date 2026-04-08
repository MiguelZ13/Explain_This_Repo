CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,   -- matches FAISS index position exactly
    text        TEXT    NOT NULL,
    file_path   TEXT,
    language    TEXT,
    chunk_type  TEXT,
    name        TEXT,
    docstring   TEXT,
    start_line  INTEGER,
    end_line    INTEGER,
    chunk_index INTEGER,
    chunk_total INTEGER,
    extra_json  TEXT    NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS repos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_name   TEXT    NOT NULL,
    repo_url    TEXT    NOT NULL,
    indexed_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_path  ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_language   ON chunks(language);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON chunks(chunk_type);
