# Phase 2: documents, retrieval and citations

Status: plan, written before any Phase 2 code. Decisions marked OPEN are made by measurement or by
the owner, not by assumption. Nothing here is a result.

## Goal

A question about the company's documents (or about the data and the documents together) is answered
with passages the reader can open, and every number taken from a document is checked by code against
the exact characters it cites. A document number the passage does not contain is flagged. The
verifier says nothing about whether the document is right (ARCHITECTURE section 4.4).

## What exists today

`passage` claims verify that a number appears in a cited span of a passage, with the same unit and a
compatible precision (`Verifier._passage`). The passage text, however, comes from the answer file
(`AnswerDoc.passages`), which in a live run is written by the model. **A model that writes both the
passage and the claim would verify against its own words.** Phase 2 closes that: passages are
looked up in a store by id, and the stored text must hash to the recorded value.

## Slices

Each slice ends in a reviewable bundle with its own tests, as in Phase 1.

| Slice | Scope | Needs a model | Done when |
|---|---|---|---|
| 2a | Document store, chunker with character spans, synthetic corpus, Postgres full-text search, stored-passage verification, retrieval metrics on dev questions | No | A passage claim verifies only against stored text; metrics for keyword search are produced by one command |
| 2b | pgvector, embedder interface, vector search, reciprocal-rank fusion | Yes (measured) | Hybrid recall measured against keyword-only on the same questions |
| 2c | Reranker interface and measurement | Yes (measured) | Reranked vs unreranked metrics reported; model choice recorded in ADR-0005 |
| 2d | Injection handling: hostile documents, delimiting, tests | Partly | Injection tests pass; no document text is ever treated as an instruction |
| 2e | Router agent (data / documents / both), mixed answers, label binding for passage numbers | Yes | Mixed questions answered with passage citations, verified |

Order is fixed because each slice's tests need the one before. 2a needs no model choice and no
network, so it can be built and tested in full here.

## Design decisions (2a)

1. **Store.** Schema `docs`, separate from `analytics`. Tables `documents` (id, title, source,
   content_sha256, text) and `chunks` (id, document_id, ordinal, char_start, char_end, text,
   text_sha256, tsv). The guarded SQL path for numbers keeps its allow-list
   (`analytics.*` only): the model can never query `docs` through it. Documents are reached only
   through the retrieval functions.
2. **Spans.** A chunk is a contiguous slice of the stored document text, `text[char_start:char_end]`.
   A citation's span is relative to the chunk text (same as the existing `PassageRef`) and the
   store can map it to absolute document offsets. Text is stored exactly as ingested apart from
   line-ending normalisation, so a span always points at the characters the reader sees.
3. **Verification against the store.** `verify_stored_passages` resolves `passage_id` to a stored
   chunk, recomputes the chunk's SHA-256 and compares it with the recorded one, and only then runs
   the existing number-in-span check. A passage text supplied by the answer file is ignored in this
   mode; a mismatch or unknown id is `UNSUPPORTED`.
4. **Keyword search first.** Postgres full-text search (`websearch_to_tsquery`, `ts_rank_cd`) over
   chunks. It is the baseline every later slice must beat on the same questions.
5. **Retrieval metrics.** Recall@k, MRR and nDCG@k over labelled questions. Labels are anchored to a
   document id and an answer string that must occur in the document, not to chunk ids, so a change
   of chunking does not invalidate them. A label whose string is not in the document fails a test.
6. **Dev and held-out.** Retrieval questions live in `eval/doc_questions/dev.yaml` and
   `heldout.yaml`. The held-out rule of `docs/HELDOUT.md` applies unchanged: nothing that tunes
   chunk size, weights, query rewriting, prompts or model choice may read the held-out file, and a
   test checks it.
7. **Corpus.** Two sources. (a) A synthetic corpus written for the retail dataset, committed:
   policies, supplier notes, product memos that state checkable numbers, with deliberately
   conflicting, stale and hostile documents for tests. (b) The owner's own documents, ingested
   locally with `va docs ingest`; those are never committed by this project and never sent to a
   hosted model without an explicit flag.
8. **No keys, no network in tests.** Embedding and reranker calls (2b, 2c) go through the router
   with mocked HTTP and cassettes, as in Phase 1. A cassette is not committed before review.

## Open decisions (made by measurement)

- Embedding model and reranker (ADR-0005). Candidates to measure, to be re-checked against each
  provider's current documentation and free-tier limits at that time, not assumed now: a small and
  a large open embedding model run locally, and a hosted free-tier embedding endpoint. The choice
  is made on dev-question metrics, with latency and cost recorded, then scored once on the held-out
  questions.
- Chunk size and overlap: tuned on dev questions only.
- Whether passage numbers need label binding as strict as cell numbers (slice 2e). Today a passage
  number is checked for presence, unit and precision, not for the entity or metric it is about.
  That is a known gap, and it is a verifier change that needs its own adversarial set.

## Risks, stated now

- A synthetic corpus written by us tests our own assumptions. Retrieval numbers on it say little
  about real documents; the owner's documents and a separately written held-out set are the
  counterweight, and no accuracy claim is made from the synthetic set alone.
- Hostile documents can only be tested against the hostile documents we think of.
- `VERIFIED_QUOTE` still means only "the text was there".

## Slice 2a as built

- `src/verified_analyst/documents/`: `chunker.py` (pure), `store.py` (ingest, search, stored
  passages), `metrics.py` (pure), `schema.sql`. Commands: `va docs ingest | search | eval`.
- Passage ids are `doc-key#ordinal.sha10`. `va verify` resolves them from the store, recomputes the
  chunk hash, checks the chunk against its document, and **ignores any passage text written in the
  answer file** (it says so on stderr). A forged passage, an edited document, an id for text that
  has since changed, and a chunk altered behind the store are all `UNSUPPORTED`; each has a test.
  Two of those tests fail when the store lookup is removed.
- The SQL guard still allows `analytics.*` only; tests prove model SQL cannot name `docs.*`.
  The read-only role can read `docs` and cannot write to it.
- Keyword search matches any content word of the question and ranks by cover density
  (`mode=any`). The all-words mode (`mode=all`) is kept for quoted phrases and returns nothing for
  most sentence questions.

### Measured here, with the caveats that matter

Mechanics check on the 21 author-written dev questions over the 14-document synthetic corpus, run
in the build container (20 scored, 1 with no answer in the corpus). Chunk size 800 characters.

| Text-search config | Mode | recall@1 | recall@3 | recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| english | any | 0.90 | 0.95 | 1.00 | 0.93 | 0.95 |
| english | all | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 |
| simple | any | 0.80 | 1.00 | 1.00 | 0.89 | 0.92 |
| simple | all | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |

What this does and does not show. The corpus is small and the questions share wording with it, so
the any-word rows sit near the ceiling and **cannot rank a better retrieval method over a worse
one**. The one clear finding is that all-words matching fails on sentence questions. The defaults
(`english`, `any`) were chosen on these dev questions, so they are tuned to them. The one question
with no answer in the corpus still returned passages, as expected from any-word matching: keyword
search has no way to say "nothing relevant". Nothing here is a claim about real documents, and the
held-out file has not been scored.

### Next

2b needs the pgvector image (compose now uses it; see docs/DOCKER_PGVECTOR_SWITCH.md), a harder and larger corpus
so that methods can be told apart, and your documents ingested locally. Both are decisions for the
owner; see the questions in the hand-over message.
