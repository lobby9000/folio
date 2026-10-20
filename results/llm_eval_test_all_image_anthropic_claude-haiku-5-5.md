# LLM extraction and the cascade

Model `claude-haiku-5-5` through `anthropic` reading the scan, mode `all`, 200 labelled receipts of the `test` split. OCR: `gray`, page mode 6. Ranker: `models/total_ranker.json`, routing threshold 0.95. Requests sent in this run: 200 (the rest came from the cache in `cache/llm`). Cost at list prices: $0.0598 for all the answers in this report, $0.0598 of it spent in this run.

## Total

| Extractor | Total correct [95% CI] |
|---|---|
| Rules (stage 2) | 151 of 200 (76%) [69%, 81%] |
| Learned ranker | 163 of 200 (82%) [76%, 86%] |
| Cascade: rules or ranker when sure, the LLM otherwise | 187 of 200 (94%) [89%, 96%] |
| LLM alone, reading the scan | 194 of 200 (97%) [94%, 99%] |

Cascade against the ranker, paired: only the cascade right on 25, only the ranker right on 1 (exact McNemar p = 8e-07).

## Where each receipt went

| Route | Receipts | Total correct |
|---|---|---|
| confirmed by cash | 63 | 62 of 63 (98%) [92%, 100%] |
| confident ranker | 48 | 43 of 48 (90%) [78%, 95%] |
| llm | 76 | 72 of 76 (95%) [87%, 98%] |
| llm for the date | 13 | 10 of 13 (77%) [50%, 92%] |

On the 76 receipts sent to the LLM, the ranker alone would have been right on 50.

## What the model answered

A value read from the scan is kept if it parses as an amount or a date.

| Status of the model's total | Answers | Right value |
|---|---|---|
| read from the image | 200 | 194 |

## Date

| Extractor | Date correct [95% CI] |
|---|---|
| Cascade: the rules' date, or the model's when it was asked | 190 of 200 (95%) [91%, 97%] |
| LLM alone | 198 of 200 (99%) [96%, 100%] |

The cascade also asks the model when the rules found no date.

## Cost

200 of 200 receipts were sent to the model.
Median time per answer 1.4 s, slowest 4.1 s.
Tokens: 535,656 in and 12,551 out in total, 2,678 in per receipt.
