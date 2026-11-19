# Phase 1a Research

Researched 2026-11-20. Every claim has a URL. "unverified" means I could not confirm it from a primary source. Free tiers change often; re-check before relying on any number here.

## 1a. Model providers

### Google AI Studio / Gemini API (free tier)

| Item | Finding | Source |
|---|---|---|
| Per-model limits (RPM/TPM/RPD) | **unverified.** The docs page gives no per-model free numbers and sends you to the AI Studio rate-limit page, which needs a login. Third-party blogs quote numbers, but they conflict and are stale. | https://ai.google.dev/gemini-api/docs/rate-limits |
| Scope of limits | Per project, not per API key. Daily quota resets at midnight Pacific. | https://ai.google.dev/gemini-api/docs/rate-limits |
| Preview/experimental models | More restrictive limits. | https://ai.google.dev/gemini-api/docs/rate-limits |
| Data use on the free tier | Unpaid services: Google may use prompts and responses to improve its products, and human reviewers may read and annotate them. The terms warn against sending sensitive, confidential or personal data. | https://ai.google.dev/gemini-api/terms |
| EEA / UK / CH | If you make API clients available to users in these regions you must use only Paid Services. | https://ai.google.dev/gemini-api/terms |
| Model churn | A third-party blog says Gemini 2.5 models shut down 2026-10-16 and that model names change often. Treat model IDs as config, never hard-code. (secondary source) | https://neuraplus-ai.github.io/blog/gemini-api-free-tier.html |

Consequence: the hosted free Gemini tier is only suitable for public or synthetic data. That fits our public datasets, but it must be a hard rule in the model router (see ADR-0003).

### Groq

| Item | Finding | Source |
|---|---|---|
| Limits (table headed "Free Plan Limits / Developer Plan Limits") | gpt-oss-120b and gpt-oss-20b: 30 RPM, 1K RPD, 8K TPM, 200K TPD. qwen/qwen3.8-27b: same. The page does not clearly separate the free plan from the developer plan, so **which column applies to a free account is unverified**; check https://console.groq.com/settings/limits. | https://console.groq.com/docs/rate-limits |
| Scope | Per organization. Whichever limit hits first applies. Cached tokens do not count. | https://console.groq.com/docs/rate-limits |
| On exceeding | HTTP 429 with a `retry-after` header (seconds). | https://console.groq.com/docs/rate-limits |
| Tool use | Supported by all hosted models. Parallel tool calls: qwen/qwen3.8-27b, llama-3.3-70b-versatile, llama-3.1-8b-instant and two others. **Not** the three gpt-oss models. JSON mode is marked supported for all listed models. | https://console.groq.com/docs/tool-use |

The 200K tokens-per-day cap is the binding constraint for an agent loop. A full eval run will not fit in one day on one model.

### OpenRouter free models (`:free`)

| Item | Finding | Source |
|---|---|---|
| Limits | 20 requests/minute. 50 requests/day if you have bought under 10 credits, 1000/day if at least 10. Based on all-time purchases. | https://openrouter.ai/docs/api-reference/limits |
| Exceeding | HTTP 429 with `rate_limit_exceeded` and `X-RateLimit-*` headers. | https://openrouter.ai/docs/api-reference/limits |
| Do failed requests count? | **unverified** (page does not say). | https://openrouter.ai/docs/api-reference/limits |
| Which `:free` models exist today, and their data policies | **unverified.** Not researched; the list changes frequently. | n/a |

50 requests per day with no purchase is too low to be a primary provider. It is a fallback only.

### Local Ollama on a 4 GB RTX 3050

All sources here are secondary (blogs), not benchmarks. Treat as directional.

| Item | Finding | Source |
|---|---|---|
| What fits | qwen2.5-coder:3b is 1.9 GB. qwen3.5:2b is 2.7 GB with tool calling. qwen2.5-coder:7b needs 4.7 GB, more than the card, so it runs partly on CPU. Budget about 3 GB usable. | https://localaimaster.com/vram/best-ollama-models-4gb-vram |
| qwen3:4b | Listed in Ollama's library as a realistic starting point for 4 GB. | https://meattacker.github.io/local-ai-model-recommender/best-local-ai-models-4gb-vram.html |
| Tool-calling quality | A Docker evaluation found Qwen 3 14B and 8B the best local tool callers. Both exceed 4 GB. | https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/ |
| Small-model tool-calling reliability | A test on an RTX 3050 6GB reported native tool-calling rates as low as 12.5% for qwen2.5-coder:0.5b. Larger small models were not captured in my excerpt. | https://rcconsult.github.io/ppxai/1.17.3/ollama-limitations |
| SQL accuracy of big local models | One 2025 self-hosted experiment got 46-54% on its own question set with 14B-32B models. Not comparable to BIRD, but a warning that small models are weak at SQL. | https://blog.nilenso.com/blog/2025/05/27/experimenting-with-self-hosted-llms-for-text-to-sql/ |
| License of Qwen3 small models | Apache 2.0 per one community Ollama page. **unverified** against the official model card. | https://ollama.com/oamazonasgabriel/qwen3.5-4b:q3_k_s-8gbGPU |
| Ollama has no rate limits or terms on local use | Reasonable by construction, but I did not find a page that states it. **unverified.** | n/a |

Conclusion: a 4 GB card can run a 2B-4B model. That is enough for routing, claim extraction and intent classification, not for reliable multi-step SQL agents. Its job is privacy fallback and offline demo, and the eval will measure how bad it is rather than assume.

## 1b. Database and hosting

| Service | Limits | On idle / exhaustion | Source |
|---|---|---|---|
| Neon Free Postgres | 100 projects, 100 CU-hours/project/month, 1 GB storage per project (20 GB total), 5 GB egress/month, 10 branches | Compute suspends after 5 minutes idle and this cannot be disabled. Cold-start time is **unverified** (not stated). CU-hours or egress exhausted: compute suspended until next period, connections fail with SQLSTATE 53000. Storage exhausted: reads work, writes that grow storage fail. No data deleted. | https://neon.com/docs/introduction/plans |
| Supabase Free | 2 free projects, 500 MB database, 5 GB egress | May pause projects with low activity in a 7-day period; restore from the dashboard. Behaviour on exceeding quotas on Free is **unverified**. | https://supabase.com/docs/guides/platform/billing-on-supabase , https://supabase.com/docs/guides/platform/going-into-prod |
| Render free web service | 750 instance-hours/month per workspace; no persistent disk; one instance | Spins down after 15 minutes without traffic; wake takes about a minute; local files lost on spin-down. Out of hours: all free web services suspended until next month. | https://render.com/docs/free |
| Render free Postgres | 1 GB | **Expires 30 days after creation**, 14-day grace, then deleted. Not usable for a long-lived demo. | https://render.com/docs/free |
| Hugging Face Spaces | Docs: CPU Basic is 2 vCPU, 16 GB RAM, free; sleeps after 48 hours idle | Docs also say creating a Docker/Gradio Space "requires a paid plan"; forum threads from 2026 report free CPU-basic quota of 0 and staff saying it needs PRO. Docs and enforcement disagree. **Do not depend on it.** | https://huggingface.co/docs/hub/en/spaces-overview , https://discuss.huggingface.co/t/free-cpu-basic-quota-is-0-on-new-account/180764 |
| Google Cloud Run | Pricing page: 240,000 vCPU-seconds and 450,000 GiB-seconds free per month; marketing page says two million requests free per month; scales to zero | Scale to zero means cold starts (duration **unverified**). Requires a billing account; whether a card is mandatory for the free tier is **unverified**. | https://cloud.google.com/run/pricing , https://cloud.google.com/run/ |
| Fly.io, Koyeb, Railway, Vercel, others | **Not researched.** | n/a | n/a |

## 1c. Datasets

Licence text below is quoted from the pages I could read. Where I could only read a licence name, I say so; fetching the canonical legal text is a follow-up item.

| Dataset | Domain | Licence | Fit for public repo + demo |
|---|---|---|---|
| UCI Online Retail | Sales + customers (541,909 rows, UK online retailer, 2010-12-01 to 2011-12-09, columns InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country) | "Creative Commons Attribution 4.0 International" (CC BY 4.0). Page text: "This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given." https://archive.ics.uci.edu/dataset/352/online+retail | Yes, cleanest licence. Single flat table; we normalise it into customers/products/invoices. |
| Olist Brazilian E-Commerce | Orders, items, payments, reviews, customers, sellers (about 100k orders, 2016-2018, 8-9 CSVs) | Kaggle metadata field: "CC BY-NC-SA 4.0". https://www.kaggle.com/api/v1/datasets/view/olistbr/brazilian-ecommerce . Non-commercial and share-alike are confirmed by a secondary source: https://scripts-and-tables.github.io/daa/data/ | Technically fine for a non-commercial portfolio, but NC blocks any commercial use of the demo and SA applies to the data. **Needs your call** (open question 2). Richest multi-table schema; reviews give text for doc/RAG tests (Portuguese). |
| AdventureWorks (Microsoft sample) | Sales, customers, products, HR | The `microsoft/sql-server-samples` repo licence is MIT: "Permission is hereby granted, free of charge, to any person obtaining a copy". https://github.com/microsoft/sql-server-samples/blob/master/license.txt . Whether the AdventureWorks data files themselves carry the same licence is **unverified.** | Candidate, but verify the data licence first. |
| Chinook | Digital media store | **unverified** (page did not show the licence name). https://github.com/lerocha/chinook-database | Not proposed until verified. |
| Bitext customer support | Support Q&A, 26,872 rows, 10 categories, "hybrid synthetic" | `cdla-sharing-1.0` (identifier only; full text not retrieved). https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset | Usable for the support slice and for doc-style text. Synthetic and generic; no link to the sales data. Sharing clause needs reading before we redistribute. |

### Document corpus for RAG

- **Recommended: synthetic corpus tied to the chosen sales dataset.** Policies, monthly business-review memos, return/support playbooks and metric glossaries generated from templates plus an LLM, built from a **ground-truth fact table** so each document's claims can be graded. Deliberately include (a) facts consistent with the data, (b) facts that conflict with the data, (c) stale facts, and (d) a few documents with embedded prompt-injection text for security tests. This is defensible because nothing is passed off as real and every fact has a known answer.
- **Real public option: CUAD** contracts. Licence listed as "CC BY 4.0", 510 contracts, more than 13,000 labels across 41 clause types. https://www.atticusprojectai.org/cuad . Realistic for RAG, but does not pair with retail data. The page did not name the source of the contracts. Use only for a separate RAG-quality test if wanted.
- **SEC EDGAR filings:** the developer page states a limit of 10 requests per second. Whether filings can be reused freely was **not confirmed** from that page. https://www.sec.gov/about/developer-resources

### Public text-to-SQL benchmarks

| Benchmark | Licence | Notes | Source |
|---|---|---|---|
| BIRD | Page: "we now modify the license of our data to `CC BY-SA 4.0`" | Dev set 1,534 examples. Metric: execution accuracy (EX); R-VES for efficiency. Test set needs submission by email, so independent scoring on test is possible but slow. Leaderboard top at time of reading: 83.40% test EX (with oracle knowledge); human baseline 92.96%. | https://bird-bench.github.io/ |
| Spider 1.0 | Site: "distributed under the CC BY-SA 4.0 license." The GitHub repo shows Apache-2.0 for its code. | 10,181 questions, 200 databases. Official metric: test-suite accuracy. Spider 1.0 no longer accepts test submissions or leaderboard updates. Snowflake argues Spider-style benchmarks are too easy for BI work. | https://yale-lily.github.io/spider , https://github.com/taoyds/spider , https://snowflake.com/en/engineering-blog/cortex-analyst-text-to-sql-accuracy-bi |

Recommendation: score on the **BIRD dev set** (public labels, runs locally, EX metric), report it clearly as dev-set EX, and treat a test-set submission as optional. Use a small Spider subset as a cheap smoke benchmark. Neither measures our actual differentiator, number verification, so the in-house gold set matters more (see ARCHITECTURE section 6).

## 1d. Competitors

| Product | What it does | Number verification | Published accuracy | Source |
|---|---|---|---|---|
| Vanna 2.0 | Open-source (MIT) chat with SQL databases; agent with a tool registry (e.g. a run-SQL tool), streams SQL, table, Plotly chart, summary | None described beyond per-user permissions, row-level security, audit logs and rate limits | None | https://github.com/Vanna-AI/vanna . **Repo archived by owner on 2026-03-29, read-only.** |
| WrenAI | Open-source GenBI engine (core Apache-2.0, some features commercial) with a semantic layer (MDL) over 20+ sources | Dry-plan validation before execution, row limits, value profiling, structured errors. Not number-level checking of answer text. A "golden eval runner" is listed as upcoming. | None | https://github.com/Canner/WrenAI |
| Dataherald | Open-source (Apache-2.0) NL-to-SQL engine with API, admin console, Slackbot | Not described | None on the page. Maintenance status unclear. | https://github.com/Dataherald/dataherald |
| Snowflake Cortex Analyst | Commercial, inside Snowflake; agentic text-to-SQL driven by a semantic model and human-approved "verified queries" | Verified-query retrieval and evaluation against verified queries; not described as checking numbers in prose | Claims "90%+" SQL accuracy on real-world cases (vendor claim, own benchmark) | https://snowflake.com/en/engineering-blog/cortex-analyst-text-to-sql-accuracy-bi , https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst-evaluations |

Not researched: Databricks Genie, Defog/SQLCoder, Metabase AI, Chat2DB and others. Treat the table as a sample, not a survey.

Take-away, limited to the pages I read: all four verify the **SQL** or its permissions; none describe checking each number in the final natural-language answer against executed results. Only Cortex Analyst has a published accuracy claim, and it is a vendor figure. The open-source tools publish none. That is the gap this project targets, but "none do this" is **unverified** beyond these four.
