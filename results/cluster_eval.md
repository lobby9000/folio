# Receipt layouts without labels

Receipts are grouped from their OCR, never from their labels; the merchant names in the SROIE labels only score the groups (branches of a few chains and label typos merged, see `folio/cluster.py`). OCR: `gray`, page mode 6. Ranker: `models/total_ranker.json`. Model answers: `results/llm_eval_test_all_image_anthropic_claude-haiku-5-5.csv`.

## Choosing a representation on the dev and train receipts

426 receipts from 165 merchants. Best threshold for each representation, by adjusted mutual information (AMI: 1 is a perfect match with the merchants, 0 is chance).

| Representation | Threshold | AMI | ARI | Groups | Homogeneity | Completeness |
|---|---|---|---|---|---|---|
| OCR text, words | 0.8 | 0.900 | 0.876 | 176 | 0.983 | 0.948 |
| first rows, characters (chosen) | 0.7 | 0.909 | 0.874 | 182 | 0.986 | 0.951 |
| word positions | 0.4 | 0.576 | 0.456 | 209 | 0.938 | 0.817 |
| thumbnail | 0.4 | 0.477 | 0.247 | 209 | 0.828 | 0.813 |

## Test: the 200 held-out receipts, grouped alone

`first rows, characters`, cut at cosine distance 0.7, fitted on the test receipts only.

| Measure | Value |
|---|---|
| Groups found / merchants | 102 / 96 |
| AMI / ARI | 0.916 / 0.897 |
| Homogeneity / completeness | 0.987 / 0.967 |
| Pairs put together that share a merchant | 94.4% |
| Pairs sharing a merchant that were put together | 86.0% |

Largest groups: 19 receipts (GARDENIA BAKERIES 19); 13 receipts (UNIHAKKA INTERNATIONAL 13); 12 receipts (RESTORAN WAN SHENG 10, RESTORAN DE COFFEE O 1, 10 GRAM GOURMET SBN 1); 8 receipts (MR D I Y 8); 8 receipts (SANYU STATIONERY SHOP 8).

## Known and new layouts

142 of the 200 test receipts come from a merchant that also has dev or train receipts. A receipt counts as a known layout when its nearest dev or train receipt is closer than 0.7. On the dev and train receipts, leaving each one out, that rule calls 298 layouts known (285 rightly) and 128 new (105 rightly). On the test receipts it agrees with the merchant labels on 171 of 200.

| Extractor (total) | Merchant seen in dev or train | New merchant | Known layout, no labels | New layout, no labels |
|---|---|---|---|---|
| Rules (stage 2) | 114 of 142 (80%) [73%, 86%] | 37 of 58 (64%) [51%, 75%] | 119 of 149 (80%) [73%, 86%] | 32 of 51 (63%) [49%, 75%] |
| Ranker (stage 3) | 120 of 142 (85%) [78%, 90%] | 43 of 58 (74%) [62%, 84%] | 127 of 149 (85%) [79%, 90%] | 36 of 51 (71%) [57%, 81%] |
| Haiku reading the scan (stage 3) | 137 of 142 (96%) [92%, 98%] | 57 of 58 (98%) [91%, 100%] | 144 of 149 (97%) [92%, 99%] | 50 of 51 (98%) [90%, 100%] |

## Honest folds for the ranker

Five-fold cross-validation of the stage 3 ranker on the dev and train receipts, three shuffles, holding out random receipts or whole layouts.

| Held out together | Accuracy, mean over folds |
|---|---|
| random receipts (stage 3) | 80.6% ± 3.4 |
| whole merchants | 80.5% ± 4.4 |
| whole groups found without labels | 80.4% ± 5.4 |

With whole merchants held out, accuracy by how many dev and train receipts the merchant has:

| Receipts from the merchant | Ranker right |
|---|---|
| 1 | 85 of 118 (72%) [63%, 79%] |
| 2 to 4 | 58 of 83 (70%) [59%, 79%] |
| 5 or more | 198 of 224 (88%) [84%, 92%] |
