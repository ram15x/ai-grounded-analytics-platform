# Phase 2b assumptions: embedding and rerank providers

Nothing in this file has been tried against a live service. Every request shape was read from the
provider's documentation page on 2026-11-20 and is built into `documents/embed_http.py` and
`documents/rerank.py`, with tests on mocked HTTP only. Rows move from "read from docs" to "seen in
a real run" only when you paste real probe output. Free-tier limits and prices were not found for
Gemini, OpenRouter or Cohere and are not assumed.

| # | Assumption | Source | Status |
|---|---|---|---|
| 1 | Gemini `models/{model}:batchEmbedContents` takes `requests[]` of `{model, content.parts[].text, output_dimensionality}` and `x-goog-api-key` | Gemini embeddings page | read from docs, not run |
| 2 | Gemini response is `embeddings[].values` | SDK examples (the REST body was not shown) | UNVERIFIED |
| 3 | `gemini-embedding-001` takes `taskType` RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY and needs manual L2 normalisation below 3072 dims | Gemini page | read from docs, not run |
| 4 | `gemini-embedding-2` has no `taskType`; the task goes in the text (`task: search result \| query: ...`, `title: none \| text: ...`) and vectors are normalised when truncated | Gemini page | read from docs, not run |
| 5 | OpenRouter `POST /api/v1/embeddings` takes `{model, input[], encoding_format}` and returns `data[].embedding` with `index` | OpenRouter page | read from docs, not run |
| 6 | OpenRouter accepts a `dimensions` field | not documented on that page | UNVERIFIED; off by default (`send_dimensions`) |
| 7 | Ollama `POST /api/embed` takes `{model, input[], truncate, dimensions}` and returns `embeddings[]`; `truncate: false` makes an over-long chunk an error | Ollama page | read from docs, not run |
| 8 | Ollama listens on 127.0.0.1:11434 | Ollama default | UNVERIFIED on this machine |
| 9 | Jina `https://api.jina.ai/v1/rerank` takes `{model, query, documents, top_n, return_documents}` | Jina page | read from docs, not run |
| 10 | Jina response is `results[].index` and `relevance_score` | common rerank shape; the page did not show it | UNVERIFIED |
| 11 | Jina free key allows 100 requests and 100K tokens per minute | Jina page | read from docs; the probe paces to 0.7 s per request |
| 12 | Model names in `config/retrieval.yaml` exist today | provider pages | UNVERIFIED; a wrong name stops the run on the first 404 |
| 13 | `BAAI/bge-reranker-v2-m3` loads in `sentence-transformers` | model card, not read | UNVERIFIED |

A wrong shape shows up as a `malformed` stop on the first call, which is the point of probing with a
small cap. The clients never retry, so a failure is visible.
