# Design data for the guard against misread digits

425 dev and train receipts, out-of-fold ranker scores. The cascade trusts the total on 270 of them (cash check or ranker score at least 0.95) and is wrong on 11: {'misread digit': 11}. The test split is not read here.

| Receipt | Route | Rule | Answer | Label | Word confidence | Second pass read it |
|---|---|---|---|---|---|---|
| 070 | cash | last total line | 141.58 | 141.50 | 68 | no |
| 247 | ranker | ranker | 733.70 | 758.70 | 89 | yes |
| 324 | cash | cash minus change | 28.96 | 28.90 | 73 | yes |
| 020 | ranker | ranker | 64.50 | 54.50 | 53 | no |
| 043 | cash | cash minus change | 184.00 | 190.00 | 71 | no |
| 071 | cash | total confirmed by cash minus change | 17.76 | 17.70 | 72 | yes |
| 146 | cash | last total line | 72.95 | 72.93 | 93 | yes |
| 149 | cash | total plus rounding | 26.80 | 26.82 | 100 | yes |
| 210 | cash | cash minus change | 638.80 | 7838.80 | 0 | no |
| 213 | ranker | ranker | 740.45 | 10.45 | 31 | no |
| 238 | cash | cash minus change | 50.15 | 12.15 | 60 | yes |

| Signal that sends a trusted total back to the LLM | Errors caught | Right answers sent back |
|---|---|---|
| OCR word confidence below 50 | 2 of 11 | 24 |
| OCR word confidence below 70 | 5 of 11 | 51 |
| OCR word confidence below 80 | 8 of 11 | 75 |
| OCR word confidence below 90 | 9 of 11 | 123 |
| the second OCR pass (`otsu`) did not read it | 5 of 11 | 16 |
| the cash check confirmed its own fallback | 4 of 11 | 17 |
| either of the last two (the guard) | 7 of 11 | 32 |
