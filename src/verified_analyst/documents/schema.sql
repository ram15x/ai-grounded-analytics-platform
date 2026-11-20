-- Document store. `va docs ingest` creates this if it is missing and never drops it.
CREATE SCHEMA IF NOT EXISTS docs;

CREATE TABLE IF NOT EXISTS docs.documents (
    doc_key        text PRIMARY KEY,
    title          text NOT NULL,
    source         text NOT NULL,
    content_sha256 text NOT NULL,
    text           text NOT NULL,
    ingested_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS docs.chunks (
    doc_key       text NOT NULL REFERENCES docs.documents (doc_key) ON DELETE CASCADE,
    ordinal       integer NOT NULL,
    char_start    integer NOT NULL,
    char_end      integer NOT NULL,
    text          text NOT NULL,
    text_sha256   text NOT NULL,
    tsv_english   tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    tsv_simple    tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
    PRIMARY KEY (doc_key, ordinal),
    CHECK (char_start >= 0 AND char_end > char_start)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_english ON docs.chunks USING gin (tsv_english);
CREATE INDEX IF NOT EXISTS chunks_tsv_simple ON docs.chunks USING gin (tsv_simple);
