# Learned extraction of the total

A ranker scores every amount OCR found on a receipt and answers with the best one (`folio/learn.py`). It is fitted on the `dev` and `train` splits, chosen by cross-validation there, and scored once on the `test` split, the same 200 labelled receipts the stage 2 rules were scored on. OCR: `gray` preprocessing, page mode 6.

## Model choice, by cross-validation on the fit pool

5 folds over 425 labelled receipts, repeated with 3 shuffles, whole receipts held out. Accuracy = share of held-out receipts whose top-scoring amount is the labelled total.

| Model | Mean accuracy | Range over the 15 folds |
|---|---|---|
| logistic (chosen) | 80.6% | 74% to 88% |
| boosting | 80.5% | 75% to 88% |

For reference, the stage 2 rules find the total on 237 of 326 (73%) [68%, 77%] of the `train` receipts, which nobody had looked at when the rules were written.

## Held-out test, scored once

| Extractor | Total correct [95% CI] |
|---|---|
| Rules (stage 2) | 151 of 200 (76%) [69%, 81%] |
| Learned ranker (logistic) | 163 of 200 (82%) [76%, 86%] |
| Ceiling: the total is among the amounts OCR produced | 176 of 200 (88%) [83%, 92%] |

Paired on the same receipts: only the ranker right on 15, only the rules right on 3 (exact McNemar p = 0.0075).

## Confidence

Test receipts sorted by the ranker's score. Keeping only the most confident ones shows how much a confidence threshold can trade coverage for accuracy.

| Most confident share | Receipts | Correct | Lowest score kept |
|---|---|---|---|
| 50% | 100 | 98 of 100 (98%) [93%, 99%] | 0.98 |
| 70% | 140 | 129 of 140 (92%) [86%, 96%] | 0.83 |
| 80% | 160 | 145 of 160 (91%) [85%, 94%] | 0.67 |
| 90% | 180 | 154 of 180 (86%) [80%, 90%] | 0.25 |
| 100% | 200 | 163 of 200 (82%) [76%, 86%] | 0.00 |

## Routing for stage 3

The cascade keeps the rules' total when it matches cash minus change, keeps the ranker's when its score is at least 0.95 (a threshold chosen on the fit pool), and sends the rest to an LLM.

| Route | Test receipts | Total correct without the LLM |
|---|---|---|
| Confirmed by cash minus change | 71 | 69 of 71 (97%) [90%, 99%] |
| Confident ranker | 53 | 46 of 53 (87%) [75%, 93%] |
| To the LLM | 76 | 50 of 76 (66%) [55%, 75%] |

## What the logistic model weighs most

Standardised weights: the change in log-odds of being the total for one standard deviation of the feature. Fitted on the whole pool.

| Feature | Weight |
|---|---|
| `value_rank` | -1.68 |
| `rules_value` | +1.43 |
| `is_round` | +1.29 |
| `equals_paid` | +1.00 |
| `value_on_payment_line` | +0.90 |
| `has_paid` | -0.76 |
| `repeats` | +0.63 |
| `last_on_row` | +0.55 |
| `value_on_total_line` | +0.43 |
| `kw_cash` | -0.41 |
| `kw_change` | -0.41 |
| `rules_row` | -0.37 |

## Every miss of the ranker on the test receipts

| Receipt | Label | Ranker | Score | Rules | Reachable | Row used |
|---|---|---|---|---|---|---|
| 000 | `9.00` | 9.60 | 0.50 | 9.60 | yes | `Round: :d Total (RM): 9.60` |
| 003 | `80.90` | 100.00 | 0.07 | 50.90 | yes | `CASH : 100.00` |
| 005 | `31.00` | 34.00 | 0.41 | 34.00 | no | `Plastic 2 18.50 34.00` |
| 008 | `112.45` | 112.46 | 0.37 | 112.46 | no | `Total (RM) : 112.46` |
| 031 | `75.00` | 78.00 | 0.96 | 78.00 | yes | `Total Sales Incl GST 78.00` |
| 040 | `343.95` | 296.00 | 0.28 | (none) | no | `2 SUB TOTAL 296.00` |
| 074 | `102.00` | 702.00 | 0.99 | 702.00 | yes | `G008OR5 ¥ 17.00 702.00 SR` |
| 102 | `73.30` | 13.30 | 0.84 | 13.30 | yes | `Total Rowiided 13.30` |
| 107 | `48.15` | 50.00 | 0.05 | 48.15 (right) | no | `CASH : 50.00` |
| 169 | `RM29.30` | 30.30 | 0.43 | (none) | no | `cash RM30,30` |
| 230 | `8.50` | 6.00 | 0.47 | (none) | no | `SR ms 6.00 G46` |
| 231 | `119.70` | 42.30 | 0.01 | 119.70 (right) | no | `9088524 1% 2.30 42.30` |
| 236 | `14.85` | 14.05 | 0.98 | 14.05 | no | `NET TOTAL 14.05` |
| 250 | `104.00` | 404.00 | 0.97 | 404.00 | no | `Subtotal : 404.00` |
| 252 | `12.37` | 27.65 | 0.14 | 0.49 | no | `Chanae RK 27.65` |
| 255 | `94.00` | 4.00 | 0.67 | 4.00 | no | `CASH 4.00` |
| 264 | `118.35` | 1128.25 | 0.95 | 1128.25 | no | `Total Incl. GST:RM 1128.25` |
| 293 | `270.00` | (none) | 0.00 | (none) | no | `` |
| 296 | `25.60` | 3.90 | 0.54 | 3.90 | yes | `Grand Total : 3.90` |
| 303 | `30.90` | 30.30 | 0.83 | 30.30 | no | `\| Grand Total : 30.30` |
| 305 | `81.80` | 81.60 | 0.75 | 81.60 | yes | `TOTAL:MYR 81.60` |
| 307 | `46.55` | 43.90 | 0.13 | 0.01 | no | `SR @ 64 43.90 2 64` |
| 315 | `11.25` | 20.00 | 0.67 | (none) | yes | `Payment -20.00` |
| 316 | `1.38` | 1.30 | 0.07 | (none) | yes | `GST TA AMT RM1. 30` |
| 342 | `19.31` | 19.34 | 0.76 | 19.34 | no | `Total Payable: 19.34` |
| 343 | `51.88` | 32.05 | 0.12 | (none) | no | `Total 6% supplies (Inc. GST): 32.05` |
| 355 | `3.21` | 11.10 | 0.28 | 3.21 (right) | yes | `Total @% supplies: 11.10` |
| 387 | `82.80` | (none) | 0.00 | (none) | no | `` |
| 395 | `RM8.35` | 10.50 | 0.09 | 0.47 | no | `Cash RM10.50` |
| 404 | `6.85` | (none) | 0.00 | (none) | no | `` |
| 407 | `89.00` | 87.00 | 0.92 | 87.00 | no | `\| 87.00 1 87.00 §` |
| 437 | `11.10` | 13.40 | 0.91 | 13.40 | no | `Tatal Rounded 13.40 \|` |
| 544 | `11.60` | 11.66 | 0.25 | 11.66 | no | `TOTAL 11.66` |
| 575 | `137.15` | 137.45 | 0.52 | 137.45 | no | `Total Sales (Inclusive 637) RM 137.45` |
| 584 | `5.00` | 10.00 | 0.83 | 10.00 | yes | `Accepted total 10.00 RH` |
| 610 | `85.20` | 65.20 | 0.97 | 65.20 | yes | `Net Total 65.20` |
| 624 | `102.00` | 702.00 | 0.99 | 702.00 | yes | `G008OR5 ¥ 17.00 702.00 SR` |
