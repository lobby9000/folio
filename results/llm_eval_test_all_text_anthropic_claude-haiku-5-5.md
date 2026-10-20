# LLM extraction and the cascade

Model `claude-haiku-5-5` through `anthropic` reading the OCR text, mode `all`, 200 labelled receipts of the `test` split. OCR: `gray`, page mode 6. Ranker: `models/total_ranker.json`, routing threshold 0.95. Requests sent in this run: 200 (the rest came from the cache in `cache/llm`). Cost at list prices: $0.0265 for all the answers in this report, $0.0265 of it spent in this run.

## Total

| Extractor | Total correct [95% CI] |
|---|---|
| Rules (stage 2) | 151 of 200 (76%) [69%, 81%] |
| Learned ranker | 163 of 200 (82%) [76%, 86%] |
| Cascade: rules or ranker when sure, the LLM otherwise | 164 of 200 (82%) [76%, 87%] |
| LLM alone (grounded answers only) | 157 of 200 (78%) [72%, 84%] |

Cascade against the ranker, paired: only the cascade right on 4, only the ranker right on 3 (exact McNemar p = 1).

## Where each receipt went

| Route | Receipts | Total correct |
|---|---|---|
| confirmed by cash | 63 | 62 of 63 (98%) [92%, 100%] |
| confident ranker | 48 | 43 of 48 (90%) [78%, 95%] |
| llm | 76 | 49 of 76 (64%) [53%, 74%] |
| llm for the date | 13 | 10 of 13 (77%) [50%, 92%] |

On the 76 receipts sent to the LLM, the ranker alone would have been right on 50.

## What the model answered

A total is kept only if its value is printed somewhere in the OCR text.

| Status of the model's total | Answers | Right value |
|---|---|---|
| grounded | 180 | 151 |
| wrong row | 6 | 6 |
| not in the text | 3 | 0 |
| unreadable | 7 | 0 |
| no answer | 4 | 0 |

## Date

| Extractor | Date correct [95% CI] |
|---|---|
| Cascade: the rules' date, or the model's when it was asked | 162 of 200 (81%) [75%, 86%] |
| LLM alone | 162 of 200 (81%) [75%, 86%] |

The cascade also asks the model when the rules found no date.

## Cost

200 of 200 receipts were sent to the model.
Median time per answer 1.2 s, slowest 11.5 s.
Tokens: 229,216 in and 7,105 out in total, 1,146 in per receipt.
