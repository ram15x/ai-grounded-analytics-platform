# Held-out questions

`eval/questions/heldout.yaml` holds 15 questions (tiers T2 and T3, ids `H-T2-01` to `H-T3-08`). They
are for scoring only, after a prompt or model choice is frozen.

**Rule.** Nothing in that file, and no answer derived from it, may be shown to a prompt-tuning
step: not in a prompt, not as an example, not to choose between prompt variants. The `metric_query`
on each question is the intended plan and counts as part of the answer key. The dev questions
(`eval/questions/dev.yaml`) are the ones the probe uses and the ones that may inform prompt work.
The file is separate from `dev.yaml` on purpose, so that nothing which loads the dev file can pick
them up. `tests/test_heldout.py` checks that no product code or prompt names or embeds the file, that
the probe reads only `dev.yaml`, and that no held-out question is also a dev question.

**Gold values are not in the repository.** `eval/gold/t2.py` (pandas only, written separately from
`t1.py` and independent of the semantic layer) computes them from the raw file:

```powershell
python eval\gold\run_heldout.py --raw "C:\path\Online Retail.xlsx" --out heldout_values.json
```

The output name is git-ignored. Money is in exact integer price ticks; a tie at a top-N cut-off is
refused, not guessed.

**Expressible by the current catalogue** means: one `MetricQuery` returns every cell the answer
needs, possibly with a derived claim (percent change, ratio, difference) over cells of that one
result. By that reading, 13 of the 15 are expressible and 2 are not.

| Id | Question, short | Catalogue |
|---|---|---|
| H-T2-01 | gross revenue, one country | expressible |
| H-T2-02 | net revenue, one month | expressible |
| H-T2-03 | top five countries by gross revenue | expressible |
| H-T2-04 | orders in each of three months | expressible |
| H-T2-05 | distinct customers, one country | expressible |
| H-T2-06 | month with the highest cancelled revenue | expressible |
| H-T2-07 | average order value, one country | expressible |
| H-T3-01 | percent change, October to November 2011 | expressible, needs a percent_change claim |
| H-T3-02 | net revenue, December 2010 vs December 2011 (partial month) | expressible |
| H-T3-03 | net revenue, one country, July to December | expressible |
| H-T3-04 | top three products by merchandise units, one country | expressible |
| H-T3-05 | merchandise units per order | expressible, needs a ratio claim |
| H-T3-06 | gross revenue on lines with no customer number | expressible, needs a difference claim |
| H-T3-07 | share of gross revenue from the top three countries | **not expressible** |
| H-T3-08 | median order value | **not expressible** |

H-T3-07 needs a total over all countries next to the top three, which one query cannot return and no
metric provides. H-T3-08 needs a median, which the catalogue lacks.

**Checks run (2026-11-20, one run, no model involved).** With the real file loaded in a scratch
warehouse, `eval/compare_heldout.py` ran each expressible question's intended plan and compared its
result with the gold: 13 of 13 agree, none differ. The two non-expressible gold values were also
compared with a direct SQL query over the warehouse and agree. The same author wrote the gold, the
questions and the semantic layer, so this is weak evidence, not an independent benchmark. No accuracy
claim is made for any model on these questions; none has been run on them.
