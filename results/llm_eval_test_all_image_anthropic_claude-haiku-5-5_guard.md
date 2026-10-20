# LLM extraction and the cascade

Model `claude-haiku-5-5` through `anthropic` reading the scan, mode `all`, 200 labelled receipts of the `test` split. OCR: `gray`, page mode 6. Ranker: `models/total_ranker.json`, routing threshold 0.95. Requests sent in this run: 0 (the rest came from the cache in `cache/llm`). Guard on: a trusted total must also be read by a second OCR pass (`otsu`, page mode 6), and the cash check does not confirm its own fallback. Cost at list prices: $0.0598 for all the answers in this report, $0.0000 of it spent in this run.

## Total

| Extractor | Total correct [95% CI] |
|---|---|
| Rules (stage 2) | 151 of 200 (76%) [69%, 81%] |
| Learned ranker | 163 of 200 (82%) [76%, 86%] |
| Cascade: rules or ranker when sure, the LLM otherwise | 191 of 200 (96%) [92%, 98%] |
| LLM alone, reading the scan | 194 of 200 (97%) [94%, 99%] |

Cascade against the ranker, paired: only the cascade right on 29, only the ranker right on 1 (exact McNemar p = 5.8e-08).

## Where each receipt went

| Route | Receipts | Total correct |
|---|---|---|
| confirmed by cash | 49 | 49 of 49 (100%) [93%, 100%] |
| confident ranker | 48 | 46 of 48 (96%) [86%, 99%] |
| second reading disagrees | 13 | 12 of 13 (92%) [67%, 99%] |
| llm | 79 | 75 of 79 (95%) [88%, 98%] |
| llm for the date | 11 | 9 of 11 (82%) [52%, 95%] |

On the 79 receipts sent to the LLM, the ranker alone would have been right on 50.

## What the model answered

A value read from the scan is kept if it parses as an amount or a date.

| Status of the model's total | Answers | Right value |
|---|---|---|
| read from the image | 200 | 194 |

## Date

| Extractor | Date correct [95% CI] |
|---|---|
| Cascade: the rules' date, or the model's when it was asked | 191 of 200 (96%) [92%, 98%] |
| LLM alone | 198 of 200 (99%) [96%, 100%] |

The cascade also asks the model when the rules found no date.

## Cost

200 of 200 receipts were sent to the model.
Median time per answer 1.4 s, slowest 4.1 s.
Tokens: 535,656 in and 12,551 out in total, 2,678 in per receipt.
