# ADR-0005: Hybrid retrieval, reranker chosen in Phase 2

Status: Accepted (design); reranker model undecided

Context: Business documents mix exact terms (SKUs, policy names) with paraphrase.

Decision: Keyword search plus vector search, fused by reciprocal-rank fusion, then reranked by a cross-encoder. The specific embedding and reranker models are chosen in Phase 2 by measuring retrieval quality on our synthetic corpus, within free-tier or local limits.

Consequences: No model is committed now. Retrieval metrics are an exit criterion of Phase 2.
