"""Documents: ingestion with character spans, keyword search, stored passages for verification.

The `docs` schema is separate from `analytics`. The SQL guard allows `analytics.*` only, so model
SQL can never read documents; they are reached through `store.search` and `store.resolve_passages`.
"""
