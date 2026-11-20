# Gold vs semantic layer: differences

> PENDING: the real Online Retail.xlsx was not available (not in Downloads or the repo's data folder), so no real gold values exist yet. The tables below are a dry run of the pipeline on the SYNTHETIC fixture and say nothing about the real data. To produce the real report: python eval/gold/run.py --raw "<path>\Online Retail.xlsx", then python eval/compare_gold.py. Every difference there must be classified by a human before Phase 1c.

Source of gold values: **SYNTHETIC FIXTURE, dry run of the pipeline. NOT real data.**
Raw file: `mini_retail.csv` (sha256 `38f7f11e5511e2aa...`, 368 rows). Generated 2026-11-20 by `eval/compare_gold.py`.

**Gold questions are author-written** (and the same author wrote the semantic layer), so agreement is weak evidence and every difference is a finding. No metric definition was changed to make numbers agree.

Questions: 15. Expressible as a catalogue query (coverage): 11/15. Agree: 10. Differ: 1. Not expressible (no layer value to compare): 4.

| ID | Question | Gold | Layer | Status |
|---|---|---|---|---|
| T1-01 | What was total gross revenue (sales before cancellations and discounts)? | 18,096.18 | 18,096.18 | agree |
| T1-02 | How much revenue was cancelled in total? | 245.93 | 245.93 | agree |
| T1-03 | What was the total value of discounts given? | 5.00 | 5.00 | agree |
| T1-04 | What was net revenue overall? | 17,845.25 | 17,845.25 | agree |
| T1-05 | How many units were sold in total? | 3,747 | 3,744 | DIFF |
| T1-06 | How many orders were placed? | 124 | 124 | agree |
| T1-07 | How many distinct customers bought something? | 28 | 28 | agree |
| T1-08 | What was the average order value? | 145.94 | 145.94 | agree |
| T1-09 | What was average revenue per customer? | 542.51 | 542.51 | agree |
| T1-10 | What share of gross revenue was cancelled? | 0.013590 | 0.013590 | agree |
| T1-11 | What was gross revenue from merchandise only (excluding postage, fees and manual lines)? | 18,038.18 | 18,038.18 | agree |
| T1-12 | How many invoice lines were sales? | 352 | n/a | not_expressible |
| T1-13 | How many different products were sold? | 13 | n/a | not_expressible |
| T1-14 | How many invoices were cancelled? | 11 | n/a | not_expressible |
| T1-15 | What was net revenue from customers we can identify? | 14,988.56 | n/a | not_expressible |

## Differences

### T1-05: How many units were sold in total?
- Gold: 3,747; layer: 3,744; layer minus gold: -3
- Classification: **intended rule difference (declared by the author before comparing; owner to confirm)**
- Author's note: Author's plain reading counts units on every sale row. The semantic layer counts merchandise stock codes only. Expected to differ if the file has sale rows with non-merchandise codes; owner to decide which rule is right.
- Layer query, restated: Compute units sold, as a single total. Definitions: units sold: Quantity on merchandise sale lines (postage, fees and manual lines excluded). Units: units sold in count. Source: analytics.sales_lines.
- Owner decision: ______

## Not expressible as a catalogue query

- T1-12: How many invoice lines were sales? (gold 352). No catalogue metric counts invoice lines.
- T1-13: How many different products were sold? (gold 13). No catalogue metric counts distinct products.
- T1-14: How many invoices were cancelled? (gold 11). No catalogue metric counts cancelled invoices.
- T1-15: What was net revenue from customers we can identify? (gold 14,988.56). No filter operator selects rows with a non-null customer_id.
