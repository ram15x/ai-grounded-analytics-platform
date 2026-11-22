-- Vector storage. Applied only when embeddings are stored (`grounded docs probe`), never by plain
-- ingest, so keyword search keeps working on a Postgres without pgvector.
-- The extension is installed in the `docs` schema: the read-only role cannot use `public`.
CREATE EXTENSION IF NOT EXISTS vector SCHEMA docs;

CREATE TABLE IF NOT EXISTS docs.chunk_embeddings (
    doc_key     text    NOT NULL,
    ordinal     integer NOT NULL,
    model_id    text    NOT NULL,
    text_sha256 text    NOT NULL,
    dims        integer NOT NULL CHECK (dims > 0),
    embedding   docs.vector NOT NULL,
    PRIMARY KEY (doc_key, ordinal, model_id),
    FOREIGN KEY (doc_key, ordinal) REFERENCES docs.chunks (doc_key, ordinal) ON DELETE CASCADE
);
