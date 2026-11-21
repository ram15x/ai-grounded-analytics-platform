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
compatible precision (`Verifier._passage`). The passage text used to come from the answer file (`AnswerDoc.passages`),
which in a live run is written by the model. **A model that writes both the passage and the claim
would verify against its own words.** Closed in slice 2a and tightened afterwards: passages are
looked up in a store by id and the stored text must hash to the recorded value; with no store at
all, `verify_answer` marks every passage claim UNSUPPORTED (`no_passage_store`) and only a test that
passes `trust_passages_written_in_the_answer_file=True` can use the answer file's text.

## Slices

Each slice ends in a reviewable bundle with its own tests, as in Phase 1.

| Slice | Scope | Needs a model | Done when |
|---|---|---|---|
| 2a | Document store, chunker with character spans, synthetic corpus, Postgres full-text search, stored-passage verification, retrieval metrics on dev questions | No | A passage claim verifies only against stored text; metrics for keyword search are produced by one command |
| 2b | pgvector, embedder interface, vector search, reciprocal-rank fusion | Yes (measured) | Hybrid recall measured against keyword-only on the same questions |
| 2c | Reranker interface and measurement | Yes (measured) | Reranked vs unreranked metrics reported; model choice recorded in ADR-0005 |
| 2d | Injection handling: hostile documents, delimiting, tests | Partly | Injection tests pass; no document text is ever treated as an instruction (mechanics built; behaviour with a real model unverified, see docs/INJECTION.md) |
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
- Whether passage numbers need label binding as strict as cell numbers (slice 2e). Slice 2e adds a
  heuristic binder (R1 to R3 below). It is stricter than presence, unit and precision but it is not
  the semantic-layer binding that cell numbers get, and its adversarial set is small.

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

## The grown corpus (decision of the owner, 2026-11-20)

80 invented documents (the 14 hand-written ones plus 66 written by `corpus/build_synthetic.py`) and
65 dev questions. The held-out file is **unchanged and locked**: 12 questions, 1 without an answer.
It is still the same author's work and **it is not an independent benchmark**, and neither is the
dev set. What was added to make retrieval harder:

- **Near-duplicate families** that differ in one number or entity: delivery to 8 countries, 7
  supplier notes, 6 category notes, 4 quarterly reviews with tables, plus an annual summary with
  figures and a table of customer segments.
- **Superseded, draft and proposed documents** as hard negatives (old delivery, service, trade,
  stock and pricing policies; an unapproved revenue draft; a trade returns policy; a rejected 2012
  proposal).
- **14 further policies, 8 minutes, memos and FAQs, 6 unrelated documents**, and **4 more hostile
  documents** (an instruction in a pricing note, a fake passage citation with a fake "verified"
  mark, an instruction in an HTML comment, a fake "administrator mode" line). They are stored as
  plain text and the injection slice will test them.
- **Dev questions:** 12 paraphrases written to share few words with their document (checked by a
  crude overlap measure, at most 0.5), 6 near-duplicate discriminations, 5 table and 2 figure
  questions, 4 superseded-text questions, 2 hostile-document questions, and **13 no-answer questions
  (20%)**, several of them near misses such as "delivery to Brazil" while delivery notes for other
  countries exist.

Mechanical checks (`tests/test_corpus.py`): the generator reproduces the committed files; every
anchor sits in its own document and in no other; every anchor fits inside one chunk; the no-answer
share is at least 8 per 40.

**Two label collisions were found and fixed on the corpus side.** The new FAQ repeated the 28-day
return window and the £250.00 trade minimum, and the new trade returns policy said "within 14 days
of delivery", which is also the anchor of a held-out question. The new documents were reworded;
the held-out file was not touched.

### Measured on the grown corpus (dev questions, build container)

Chunk size 800 characters, so every document is one chunk (80 chunks). 52 questions with an answer
are scored; the 13 without one are not.

| Text-search config | Mode | recall@1 | recall@3 | recall@5 | recall@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|---|
| english | any | 0.60 | 0.77 | 0.85 | 0.92 | 0.70 | 0.73 |
| english | all | 0.10 | 0.13 | 0.15 | 0.15 | 0.12 | 0.13 |
| simple | any | 0.44 | 0.67 | 0.79 | 0.85 | 0.57 | 0.62 |
| simple | all | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |

By kind, `english` / `any`: lookup 16 questions, recall@5 0.94; near-duplicate 6, recall@1 0.33 and
recall@5 1.00; table 5, recall@5 1.00; paraphrase with low overlap 12, **recall@1 0.33, recall@5
0.42, MRR 0.40**. At 300 characters (146 chunks) `english` / `any` gives recall@5 0.85 and MRR 0.65.
All 13 no-answer questions still returned passages, because keyword search cannot say "nothing
relevant".

How to read this. The corpus now separates cases: keyword search is weak on paraphrases, which is
the gap that vectors and a reranker are meant to close, and good enough on exact wording. The
defaults were chosen earlier on the first dev questions, so they are not independent of this
set. These are mechanics numbers on invented text by the author, **not accuracy claims**, and the
held-out set has not been scored.


### Measured on the first, small corpus (kept for comparison; superseded by the grown-corpus section above)

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

### 2b as built

- `documents/vectorstore.py`: embeddings in `docs.chunk_embeddings` keyed by (document, chunk,
  model id); exact cosine search; the model id includes the model, dimensions and prefixes.
- `documents/fusion.py`: reciprocal-rank fusion (k=60) and a reranking wrapper (depth 20).
- `documents/embed_http.py`, `documents/rerank.py`: clients for Gemini, OpenRouter, a local CPU
  model through the optional `fastembed` extra (Ollama is not used), a
  Jina-style rerank API and a local cross-encoder. No retries; errors map to short kinds; keys are
  redacted. Request shapes come from provider docs and are untried (docs/PHASE_2B_ASSUMPTIONS.md).
- `documents/settings.py`, `config/retrieval.yaml`: the candidates. Nothing is chosen yet.
- `va docs probe`: runs keyword, vector, hybrid and hybrid+rerank on the dev questions and writes a
  report (docs/LIVE_RUNBOOK.md section 6). Hosted models see only public documents.
- Tests use fake models and mocked HTTP only; the fake models test mechanics, not quality.

No embedder or reranker has been chosen. That waits for the probe output you run. With questions and
documents from one author it will show which candidate behaves better on this set, not how any would
do on real documents. The held-out set is still locked and is not an independent benchmark.

### Fusion and abstention (dev only)

With real Gemini vectors the owner saw vector search alone ahead of the default hybrid on the dev
questions (recall@1 0.88 against 0.71, MRR 0.93 against 0.82; one run, one embedder). The default
fuses keyword and vector rankings with equal weight, RRF constant 60 and 50 candidates each.
`va docs probe --fusion-study` scores a fixed 27-setting grid (constant 10/30/60, depth 10/20/50,
keyword weight 1/0.5/0.25) plus both single methods on the same questions and reports each change
against the default with a paired bootstrap interval. It selects nothing and the default has not
been changed. A setting is worth considering only if the same direction shows up with a second
embedder. Run here with the fake embedder, the study only proves the mechanics (the fake vectors
are a word-hash, so its numbers say nothing about real embeddings).

The probe report also measures whether the top score separates answerable from no-answer
questions (AUC). Search never abstains by itself; no threshold is set.

### Held-out gate and the local reranker (slice 2c)

ADR-0009 fixes three candidates before any held-out scoring: A vector-only, B hybrid (RRF constant
10, 20 candidates, keyword weight 0.25) and the baseline default. `va docs heldout` scores only
those, once per embedder, from an ADR that is committed and unchanged; it prints per-question
differences and applies no decision rule. The shipped default is unchanged. The held-out set is
still locked to everything else and is still not an independent benchmark (same author, 12
questions). The guard tests now allow exactly one module, `heldout_run.py`, to name the file.

2c adds a local cross-encoder reranker through fastembed (feature, model names and sizes
UNVERIFIED), measured by `va docs probe --reranker <name>` as reranked against unreranked on dev
with bootstrap intervals. A hosted reranker keeps the same interface. 2d (injection handling) is below; 2e
(the router agent) is next.

### Held-out outcome (ADR-0010)

The pre-registered rule was applied once per embedder. B (hybrid, k=10, depth 20, keyword weight
0.25) fails the baseline test on both Gemini embedding-2 and bge-small (MRR below the default's; worse
on 3 and 2 questions, limit 1), so the shipped default stays. The dev finding that the default was
worse than vector-only reversed on held-out. Limits: 11 answerable questions, one author, one
corpus; the explanation (more lookup and exact-term questions in held-out) is an unverified
hypothesis. The fusion choice is **undecided**. The 12 held-out document questions are **spent**;
a new held-out set on real public documents is needed before any further retrieval decision.

Reranker (dev probe only): the local MiniLM cross-encoder raised MRR by 0.105 on the default hybrid,
had no measurable effect on vector-only or B, and lowered recall@5 from 1.00 to 0.96. No reranker
in the default pipeline; the interface stays; recheck on harder public data.

### 2d as built (injection handling)

`documents/untrusted.py`: a pattern scanner, delimited and withheld rendering for model prompts
(`render_passages`, `PASSAGE_RULES` version `passages.v1`), and a verifier rule that a passage claim
overlapping flagged text is `UNSUPPORTED` (`passage_span_flagged`). `va docs search` marks flagged
passages and prints titles safely; `va docs ingest` warns. Tests: the five hostile corpus documents
are flagged and no other corpus document is; an obedient attacker's answers are rejected while
true facts in the same chunks still verify; the prompt block never contains the injected text.
Limits are in docs/INJECTION.md: the scanner is heuristic, a quote that really is in a document
verifies as a quote, prose is not checked, and no real model has been run against the rules.

### 2e as built (router, mixed answers, label binding)

**Route.** `llm/answer.py` `answer_question` first asks the model for a route (`route.v1`:
data, documents, both, or abstain with a reason). The route prompt holds the catalogue and a fixed
operator-written description of the corpus; it holds no passage and no data. An invalid reply is
asked once more, then the question abstains `route_failed`. A route `abstain` ends the run with the
router's reason (`ambiguous`, `out_of_scope`, `forecast`, `causal`, `personal_data`). Data runs the
existing plan, compile, guard and execute steps. Documents retrieves passages (keyword by default;
any search function can be given) and skips planning. Both does both. Nothing is retrieved for a
data route.

**Passage numbers are placeholders too.** The composer never types a passage figure. The system
lists, for each rendered passage (flagged paragraphs already withheld), the numbers the extractor
finds as handles `P2 n3`, and the composer writes
`{"kind": "passage_number", "label": "P2", "number": "n3"}`. The system writes the digits from the
stored text (`documents/numbers.py`, `template.py`), builds a `PassageRef` with the span from the
stored chunk, and the verifier re-checks it: span in range, no flagged overlap, number present,
unit compatible, and the binder. Handles never include a number inside flagged text. A handle that
was not listed is `unknown_passage_number`. Decimals and scale are not allowed on a passage number.
Derived claims over passage numbers are not built.

**Label binding (`documents/binding.py`, default on, fails closed).** R1: the words of the claim
label that the passage itself uses must all sit within six words of the number in the passage, and
at least one label word must (`label_not_in_passage_window`). R2: the label words must occur in
the prose sentence that carries the number (`label_not_in_sentence`). R3: every capitalised,
non-initial word in that sentence that is not a label word must occur in the cited passage
(`entity_not_in_passage`). Stemming is a simple suffix strip. This catches a number moved to a
neighbouring subject or an entity the passage never names. It does not understand meaning: a
label that really is near a number but means something else passes, and a legitimate paraphrase
can be refused. It also applies to every passage claim, including those from `va verify`.

**Sensitivity.** Each retrieval returns the sensitivity of its documents (public only if every
document is public). The compose request carries the dataset tag `docs.public` (tagged public in
`config/router.yaml`) or `docs.internal` (untagged, so internal). A hosted provider is refused the
request, so the answer abstains `sensitivity`; a local provider is not. The route request carries
no passage text, so a refusal cannot leak any.

**Other abstains.** `no_passage`: nothing retrieved, or no passage has a quotable figure after
withholding. `unverified`: no attempt verified a figure (a documents-only answer that states no
placeholder figure is never shown as verified). A plan abstain code passes through.

**Not built.** Document facts that are words, not numbers, are not verified (docs/INJECTION.md).
The router is a prompt, not a trained component, and its prompt is a first draft. Hybrid retrieval
inside `answer_question` is possible through the search function but is not wired to a command.

**Tests (no real model).** `tests/test_answer_e2e.py` runs a FakeProvider through a mixed question,
a documents-only question, a data-only question, no-answer paths (router abstain, bad route twice,
nothing retrieved, silent composer, plan abstain), sensitivity (hosted refused, local allowed,
mixed batch counts as internal), retrieval with the fake embedder, and each of the five hostile
documents with an obedient attacker composer. `tests/test_binding.py` and
`tests/test_passage_numbers.py` cover the binder and the placeholders.

**Probe.** `va route probe` is in docs/LIVE_RUNBOOK.md section 7. Questions:
`eval/router_questions/dev.yaml` (12 router questions: 3 data, 3 documents, 3 both, 3 abstain;
plus 5 injection questions, one per hostile document) and `heldout.yaml` (12, locked, same mix).
Nothing in `src` names the locked file; a guard test checks that, that no question is shared with
any other question file, and that no prompt embeds a question from either. All labels are the
author's. No accuracy claim is made.

**After the first real-model run of the router probe.** On Groq gpt-oss-20b the route reply failed
schema validation on the documents and both routes (9 of 17 and 11 of 23 route replies valid), the
data and abstain routes matched, and no injection question reached a composer, so the injection
counts of that run say nothing about injection. The reports could not show why. Changes: per-step
reply and field errors and redacted HTTP errors in the report; `--record`; prompt `route.v2` (exact
shapes and worked examples) with a retry that shows the field errors; a report that separates a
wrong route from an invalid reply and from a later failure. The `route.v1` text is kept for
history. The run counts were not a measurement of the router's quality.

