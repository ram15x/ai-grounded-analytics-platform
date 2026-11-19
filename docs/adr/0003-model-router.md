# ADR-0003: Model router with a data-sensitivity rule

Status: Accepted (design)

Context: Free hosted tiers have tight limits (Groq 200K tokens/day on listed models, OpenRouter 50 requests/day without credits) and Gemini's unpaid terms allow training use and human review. A 4 GB local GPU only fits 2B-4B models.

Decision: A router chooses a model per task from config. Rules: (1) data marked non-public never goes to a hosted free provider; (2) on HTTP 429 honour `retry-after` and fall back along a configured chain; (3) model IDs live in config, never in code; (4) every call records provider, model, tokens, cost estimate and latency. Local small models handle routing, claim extraction and the offline path; the eval decides whether they are acceptable for SQL.

Consequences: More config and a fallback test matrix. Results depend on providers' current free limits, which must be re-checked.
