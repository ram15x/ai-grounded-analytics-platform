# Gold vs semantic layer: differences

> Real data. T1-05 was reworded after the first run (see Owner decisions).

Source of gold values: **REAL DATA**
Raw file: `Online Retail.xlsx` (sha256 `43465a06f2ccf7c8...`, 541909 rows). Generated 2026-11-20 by `eval/compare_gold.py`.

**Gold questions are author-written** (and the same author wrote the semantic layer), so agreement is weak evidence and every difference is a finding. No metric definition was changed to make numbers agree.

Questions: 15. Expressible as a catalogue query (coverage): 11/15. Agree: 11. Differ: 0. Not expressible (no layer value to compare): 4.

| ID | Question | Gold | Layer | Status |
|---|---|---|---|---|
| T1-01 | What was total gross revenue (sales before cancellations and discounts)? | 10,666,684.54 | 10,666,684.54 | agree |
| T1-02 | How much revenue was cancelled in total? | 891,116.27 | 891,116.27 | agree |
| T1-03 | What was the total value of discounts given? | 5,696.22 | 5,696.22 | agree |
| T1-04 | What was net revenue overall? | 9,769,872.05 | 9,769,872.05 | agree |
| T1-05 | How many merchandise units were sold in total (postage, fees and manual lines excluded)? | 5,576,996 | 5,576,996 | agree |
| T1-06 | How many orders were placed? | 19,960 | 19,960 | agree |
| T1-07 | How many distinct customers bought something? | 4,338 | 4,338 | agree |
| T1-08 | What was the average order value? | 534.40 | 534.40 | agree |
| T1-09 | What was average revenue per customer? | 2,054.27 | 2,054.27 | agree |
| T1-10 | What share of gross revenue was cancelled? | 0.083542 | 0.083542 | agree |
| T1-11 | What was gross revenue from merchandise only (excluding postage, fees and manual lines)? | 10,271,034.61 | 10,271,034.61 | agree |
| T1-12 | How many invoice lines were sales? | 530,104 | n/a | not_expressible |
| T1-13 | How many different products were sold? | 3,812 | n/a | not_expressible |
| T1-14 | How many invoices were cancelled? | 3,772 | n/a | not_expressible |
| T1-15 | What was net revenue from customers we can identify? | 8,300,065.81 | n/a | not_expressible |

## Differences

None in this final run. The one difference found on the first real run, and its resolution, is
recorded below.
## Not expressible as a catalogue query

- T1-12: How many invoice lines were sales? (gold 530,104). No catalogue metric counts invoice lines.
- T1-13: How many different products were sold? (gold 3,812). No catalogue metric counts distinct products.
- T1-14: How many invoices were cancelled? (gold 3,772). No catalogue metric counts cancelled invoices.
- T1-15: What was net revenue from customers we can identify? (gold 8,300,065.81). No filter operator selects rows with a non-null customer_id.

## Owner decisions (2026-11-20)

### T1-05 units sold: keep the layer rule, reword the question

First real run, with the question worded "How many units were sold in total?" and gold counting
every sale row: gold 5,588,376, layer 5,576,996, layer minus gold -11,380. The 11,380 units are
quantity on sale rows with non-merchandise codes (postage, fees, manual lines).

Decision: the layer rule stands (merchandise only). The question is reworded to "How many
merchandise units were sold in total (postage, fees and manual lines excluded)?" and its gold
function is now `merchandise_units_sold`. The metric definition was not changed. Gold recomputed
from the raw file: 5,576,996, equal to the layer.

Read this honestly: the gold was edited after the difference was seen, so T1-05's agreement now
shows only that the layer implements its own rule correctly (the gold re-derives it from raw pandas
with its own regex); it does not show the rule is the right one. That was an owner decision. The
all-rows reading is still available as `units_sold` in `eval/gold/t1.py`.

### Exact duplicates stay counted

5,268 exact duplicate rows are kept in every metric. Sensitivity (from `grounded profile-db`):

| Metric | With | Without | Change |
|---|---|---|---|
| Net revenue | 9,769,872.05 | 9,748,131.07 | -0.22% |
| Units sold | 5,576,996 | 5,561,280 | -0.28% |
| Orders, active customers | 19,960, 4,338 | 19,960, 4,338 | none |

This did not affect any gold comparison, because gold also keeps them (rule written in
`eval/questions/dev.yaml`).

## Reproduce

    grounded load --source "Online Retail.xlsx"
    python eval/gold/run.py --raw "Online Retail.xlsx"
    python eval/compare_gold.py

Gold values were produced twice, independently (build chat and CTO chat), and are identical.
