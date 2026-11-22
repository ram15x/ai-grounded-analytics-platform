-- Document store. `grounded docs ingest` creates this if it is missing and never drops it.
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

-- Sensitivity gates what may be sent to a hosted model. Fail closed: the default is internal.
ALTER TABLE docs.documents ADD COLUMN IF NOT EXISTS sensitivity text NOT NULL DEFAULT 'internal';
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'documents_sensitivity_check') THEN
        ALTER TABLE docs.documents ADD CONSTRAINT documents_sensitivity_check
            CHECK (sensitivity IN ('public', 'internal'));
    END IF;
END $$;
