# Demo records

`results.json` holds 17 question runs (the 12 router dev questions and 5 injection dev questions).
Model replies were recorded from real calls to openai/gpt-oss-20b on Groq and replayed through the
real pipeline; the SQL, result rows, passages and verdicts were recomputed on 2026-11-21 against the
UCI Online Retail warehouse. Five questions have no recorded reply and are marked `not_recorded`
(not a model abstention). Run `va serve` to view them. No accuracy claim: dev questions only.
