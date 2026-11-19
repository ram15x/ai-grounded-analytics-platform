# ADR-0002: Postgres as warehouse and store, DuckDB as analytical engine

Status: Accepted (design)

Context: We need a hosted free database for the demo, vector and full-text search, and a fast local engine for analytics and benchmark databases.

Decision: Postgres (pgvector, FTS) holds tables, documents, jobs and traces. DuckDB runs aggregate queries over Parquet exports and loads benchmark databases for eval. Both are accessed through the same SQL guard and read-only controls.

Consequences: Two SQL dialects to support. The semantic layer must compile to both or state which engine a metric targets. Neon's free tier caps storage at 1 GB per project (docs/RESEARCH.md), so the demo dataset must stay small.
