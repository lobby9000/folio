# Field extraction: total and date

Rules from `folio/extract.py`. They were written looking only at the `dev` split (100 receipts, the stage 1 sample). The `test` split (200 other receipts, drawn with seed 1) was scored once the rules were frozen. OCR: `gray` preprocessing, page mode 6, Tesseract 5.3.4. *Reference* runs the same rules on the dataset's own transcription, to tell reading errors from rule errors. Values are compared after normalising, so `RM 5.90` equals `5,90`. A receipt whose label for a field is empty is left out of that field.

## Accuracy

Correct answers out of the receipts that have a label, with 95% Wilson intervals.

| Split | Text | Method | Total | Date |
|---|---|---|---|---|
| dev | OCR | rules | 86 of 99 (87%) [79%, 92%] | 84 of 100 (84%) [76%, 90%] |
| dev | OCR | baseline | 43 of 99 (43%) [34%, 53%] | 84 of 100 (84%) [76%, 90%] |
| dev | reference | rules | 96 of 99 (97%) [91%, 99%] | 100 of 100 (100%) [96%, 100%] |
| dev | reference | baseline | 51 of 99 (52%) [42%, 61%] | 100 of 100 (100%) [96%, 100%] |
| test | OCR | rules | 151 of 200 (76%) [69%, 81%] | 161 of 200 (80%) [74%, 85%] |
| test | OCR | baseline | 77 of 200 (38%) [32%, 45%] | 161 of 200 (80%) [74%, 85%] |
| test | reference | rules | 190 of 200 (95%) [91%, 97%] | 198 of 200 (99%) [96%, 100%] |
| test | reference | baseline | 99 of 200 (50%) [43%, 56%] | 198 of 200 (99%) [96%, 100%] |

## Where the misses come from

Rules on OCR text. A miss is put down to OCR when the same rules get that receipt right on the reference text. The last two columns are ceilings: how often the right value appears anywhere in the OCR text once normalised, and how often the label appears character for character (the check used in stage 1).

| Split | Field | Correct | Missed | Missed because of OCR | Missed on the reference text too | Value anywhere in the text | Label verbatim in the text |
|---|---|---|---|---|---|---|---|
| dev | total | 86 of 99 | 13 | 11 | 2 | 94 | 90 |
| dev | date | 84 of 100 | 16 | 16 | 0 | 84 | 82 |
| test | total | 151 of 200 | 49 | 39 | 10 | 176 | 161 |
| test | date | 161 of 200 | 39 | 37 | 2 | 163 | 161 |

## Rules against the baseline

Same receipts, paired. The baseline takes the largest amount and the first date. p is an exact McNemar test on the receipts where only one of the two is right.

| Split | Text | Field | Only rules right | Only baseline right | p |
|---|---|---|---|---|---|
| dev | OCR | total | 45 | 2 | 1.6e-11 |
| dev | OCR | date | 0 | 0 | 1 |
| dev | reference | total | 46 | 1 | 6.8e-13 |
| dev | reference | date | 0 | 0 | 1 |
| test | OCR | total | 84 | 10 | 1e-15 |
| test | OCR | date | 0 | 0 | 1 |
| test | reference | total | 97 | 6 | 3e-22 |
| test | reference | date | 0 | 0 | 1 |

## Checking the total against cash minus change

Rules on OCR text, split by whether the extracted total equals cash tendered minus change.

| Split | Check | Receipts | Total correct |
|---|---|---|---|
| dev | agrees | 36 | 34 of 36 (94%) [82%, 98%] |
| dev | disagrees | 9 | 6 of 9 (67%) [35%, 88%] |
| dev | nothing to check | 54 | 46 of 54 (85%) [73%, 92%] |
| test | agrees | 71 | 69 of 71 (97%) [90%, 99%] |
| test | disagrees | 16 | 9 of 16 (56%) [33%, 77%] |
| test | nothing to check | 113 | 73 of 113 (65%) [55%, 73%] |

## Which rule gave the total

Rules on OCR text. Each answer comes from one rule of `extract_total`.

| Split | Rule | Receipts | Total correct |
|---|---|---|---|
| dev | last total line | 80 | 73 of 80 (91%) [83%, 96%] |
| dev | (no answer) | 5 | 0 of 5 (0%) [0%, 43%] |
| dev | cash minus change | 5 | 4 of 5 (80%) [38%, 96%] |
| dev | total plus rounding | 4 | 4 of 4 (100%) [51%, 100%] |
| dev | value on the next row | 3 | 3 of 3 (100%) [44%, 100%] |
| dev | card payment | 1 | 1 of 1 (100%) [21%, 100%] |
| dev | total confirmed by cash minus change | 1 | 1 of 1 (100%) [21%, 100%] |
| test | last total line | 163 | 139 of 163 (85%) [79%, 90%] |
| test | (no answer) | 18 | 0 of 18 (0%) [0%, 18%] |
| test | cash minus change | 9 | 7 of 9 (78%) [45%, 94%] |
| test | total plus rounding | 5 | 4 of 5 (80%) [38%, 96%] |
| test | value on the next row | 4 | 0 of 4 (0%) [0%, 49%] |
| test | total confirmed by cash minus change | 1 | 1 of 1 (100%) [21%, 100%] |

## Every miss of the rules on OCR text

| Split | Receipt | Field | Label | Extracted | Right on reference text | Value in the OCR text | Rule | Line used |
|---|---|---|---|---|---|---|---|---|
| dev | 014 | date | `22/12/2017` | (none) | yes | no | - | `` |
| dev | 070 | date | `05-02-18` | (none) | yes | no | - | `` |
| dev | 075 | date | `30/03/2018` | (none) | yes | no | - | `` |
| dev | 193 | date | `01/03/18` | (none) | yes | no | - | `` |
| dev | 241 | date | `25/FEB/2017` | (none) | yes | no | - | `` |
| dev | 247 | date | `6/2/2017` | 2017-02-04 | yes | no | first date | `CMINO9E79 4/2/2017 12242 EN` |
| dev | 249 | date | `27/03/2018` | (none) | yes | no | - | `` |
| dev | 265 | date | `15/05/2017` | (none) | yes | no | - | `` |
| dev | 362 | date | `23/10/2017` | (none) | yes | no | - | `` |
| dev | 400 | date | `28/02/2016` | 2014-02-28 | yes | no | date after a label | `RECEIPT W: CSOU052997 DATE: 28/02/2014` |
| dev | 414 | date | `OCT 3, 2016` | (none) | yes | no | - | `` |
| dev | 487 | date | `18/09/2017` | (none) | yes | no | - | `` |
| dev | 505 | date | `13/07/2017` | 2017-07-14 | yes | no | date after a label | `RECEIPT. #2 C8PO397200 DATE: 14/07/2017` |
| dev | 516 | date | `24/05/2018` | 2018-08-24 | yes | no | date after a label | `RECEIPT #: CS00040529 DATE: 24/08/2018` |
| dev | 545 | date | `04-04-2018` | (none) | yes | no | - | `` |
| dev | 625 | date | `27/03/2018` | (none) | yes | no | - | `` |
| dev | 070 | total | `141.50` | 141.58 | yes | yes | last total line | `Total Sales (Inclusive GST? RN 141.58` |
| dev | 095 | total | `31.45` | 3.47 | yes | no | last total line | `Fotal RM Incl. of GST 3.47` |
| dev | 247 | total | `758.70` | 733.70 | yes | yes | last total line | `Final Total Hy 733.70` |
| dev | 249 | total | `130.00` | (none) | yes | yes | - | `` |
| dev | 256 | total | `12.00` | 42.00 | yes | yes | last total line | `Total “42.00` |
| dev | 265 | total | `4.70` | (none) | yes | yes | - | `` |
| dev | 297 | total | `21.60` | 12.40 | no | yes | last total line | `Grand Total : 12.40 :` |
| dev | 317 | total | `RM5.00` | (none) | yes | no | - | `` |
| dev | 324 | total | `28.90` | 28.96 | yes | no | cash minus change | `CHANGE o.00` |
| dev | 414 | total | `RM33.90` | (none) | yes | no | - | `` |
| dev | 505 | total | `4.00` | (none) | yes | yes | - | `` |
| dev | 533 | total | `38.60` | 30.60 | yes | yes | last total line | `Total After Adj Inc]. GST 30.60` |
| dev | 561 | total | `17.22` | 17.20 | no | no | last total line | `TTL ATF RND 17.20` |
| test | 031 | date | `06/03/2018` | (none) | yes | no | - | `` |
| test | 038 | date | `11/03/2018` | (none) | yes | no | - | `` |
| test | 049 | date | `15/03/2018` | 2018-03-16 | yes | no | date after a label | `Salesperson: Date: 16/03/2018` |
| test | 109 | date | `09/02/2018` | (none) | yes | no | - | `` |
| test | 130 | date | `05/02/2018` | 2018-02-06 | yes | no | date after a label | `Doc No CS$00065374 Date 06/02/2018 \|` |
| test | 152 | date | `25032018` | (none) | no | no | - | `` |
| test | 169 | date | `2018-03-18` | 2018-03-16 | yes | no | date after a label | `Date: 2018-03-16 13:18:48` |
| test | 183 | date | `03/02/2018` | (none) | yes | no | - | `` |
| test | 230 | date | `27/03/2018` | (none) | yes | no | - | `` |
| test | 231 | date | `15-03-18` | 2013-03-15 | yes | no | first date | `15-03-13 15:41 SHO1 BOIS 12 ROOOZ43654` |
| test | 242 | date | `05 JAN 2017` | (none) | yes | no | - | `` |
| test | 250 | date | `09/04/2018` | (none) | yes | no | - | `` |
| test | 252 | date | `23/04/2017` | (none) | yes | no | - | `` |
| test | 255 | date | `25/04/2017` | (none) | yes | no | - | `` |
| test | 273 | date | `14/JUN/2017` | (none) | yes | no | - | `` |
| test | 293 | date | `6/4/2018` | (none) | yes | no | - | `` |
| test | 312 | date | `19/05/18` | 2018-08-13 | yes | no | first date | `16:58 = 13/08/18` |
| test | 337 | date | `21/08/2017` | (none) | yes | no | - | `` |
| test | 342 | date | `05/08/2017` | (none) | yes | no | - | `` |
| test | 375 | date | `01/10/2017` | 2017-01-16 | yes | no | date after a label | `V E0514 Date: 01/16/2017` |
| test | 381 | date | `(06/12/2016)` | (none) | yes | no | - | `` |
| test | 387 | date | `03-02-16` | (none) | yes | no | - | `` |
| test | 404 | date | `19/05/16` | (none) | yes | no | - | `` |
| test | 410 | date | `21/05/2018` | (none) | no | no | - | `` |
| test | 423 | date | `17/05/2018` | 2018-06-17 | yes | no | date after a label | `Date: 17/06/2018 10:54:21 PM` |
| test | 426 | date | `27/05/2018` | 2018-05-21 | yes | no | first date | `21/05/2018 13:48 1010 007 0070177` |
| test | 437 | date | `25/05/2018` | 2016-05-25 | yes | no | first date | `. \| ORD #6S -REG a1- 25/05/2016 18:12 250` |
| test | 449 | date | `28/04/2018` | (none) | yes | no | - | `` |
| test | 467 | date | `20/05/2018` | 2018-06-20 | yes | no | date after a label | `Dec.No. C$00172418 Date: 20/06/2018` |
| test | 471 | date | `10/04/2017` | (none) | yes | no | - | `` |
| test | 472 | date | `18/04/2017` | (none) | yes | no | - | `` |
| test | 548 | date | `05-05-2018` | 2015-06-05 | yes | no | date after a label | `Date : 05-06-2015 11:25:29` |
| test | 562 | date | `11-06-2018` | 2018-08-11 | yes | yes | date after a label | `Date +: 11-08-2018 12:51:34` |
| test | 566 | date | `04-06-2018` | 2000-06-04 | yes | no | date after a label | `Date : 04-06-00) h2` |
| test | 584 | date | `28/05/18` | 2018-08-28 | yes | yes | first date | `28/08/18 11241` |
| test | 603 | date | `17/04/2018` | (none) | yes | no | - | `` |
| test | 605 | date | `03/05/2018` | (none) | yes | no | - | `` |
| test | 609 | date | `11/05/2018` | (none) | yes | no | - | `` |
| test | 620 | date | `17-06-2018` | 2018-06-07 | yes | no | first date | `‘a: \|7-06-2018 14:07:29` |
| test | 000 | total | `9.00` | 9.60 | yes | yes | last total line | `Round: :d Total (RM): 9.60` |
| test | 003 | total | `80.90` | 50.90 | yes | yes | cash minus change | `Change : 49.10` |
| test | 005 | total | `31.00` | 34.00 | yes | no | last total line | `Total Amount: 34.00"` |
| test | 008 | total | `112.45` | 112.46 | yes | no | last total line | `Total (RM) : 112.46` |
| test | 016 | total | `73.00` | (none) | no | yes | - | `` |
| test | 018 | total | `73.00` | (none) | no | yes | - | `` |
| test | 031 | total | `75.00` | 78.00 | yes | yes | last total line | `Total After Adj Incl 637 78.00` |
| test | 040 | total | `343.95` | (none) | yes | no | - | `` |
| test | 047 | total | `170.00` | (none) | no | yes | - | `` |
| test | 074 | total | `102.00` | 702.00 | yes | yes | last total line | `Total Amt Rounded ——S«702.00` |
| test | 098 | total | `42.40` | (none) | yes | yes | - | `` |
| test | 102 | total | `73.30` | 13.30 | yes | yes | last total line | `Total Rowiided 13.30` |
| test | 124 | total | `5.30` | 0.30 | yes | yes | last total line | `Total Sales (Inclusive of GST) : §.30` |
| test | 152 | total | `RM41.45` | (none) | yes | yes | - | `` |
| test | 163 | total | `108.50` | 106.50 | yes | yes | last total line | `Total Sales (Inclusive G57) RM 106.50` |
| test | 169 | total | `RM29.30` | (none) | yes | no | - | `` |
| test | 181 | total | `6.00` | (none) | no | yes | - | `` |
| test | 212 | total | `29.00` | (none) | yes | yes | - | `` |
| test | 219 | total | `100.00` | (none) | yes | yes | - | `` |
| test | 230 | total | `8.50` | (none) | yes | no | - | `` |
| test | 236 | total | `14.85` | 14.05 | yes | no | last total line | `NET TOTAL 14.05` |
| test | 250 | total | `104.00` | 404.00 | yes | no | value on the next row | `Ezyment : 404.00` |
| test | 252 | total | `12.37` | 0.49 | no | no | value on the next row | `Rounding Agtusiments Re “0,49` |
| test | 255 | total | `94.00` | 4.00 | yes | no | cash minus change | `CHANGE 0.00` |
| test | 258 | total | `RM1.00` | (none) | yes | yes | - | `` |
| test | 264 | total | `118.35` | 1128.25 | yes | no | last total line | `Total Incl. GST:RM 1128.25` |
| test | 293 | total | `270.00` | (none) | yes | no | - | `` |
| test | 296 | total | `25.60` | 3.90 | no | yes | last total line | `Grand Total : 3.90` |
| test | 303 | total | `30.90` | 30.30 | yes | no | last total line | `\| Grand Total : 30.30` |
| test | 305 | total | `81.80` | 81.60 | yes | yes | last total line | `TOTAL:MYR 81.60` |
| test | 307 | total | `46.55` | 0.01 | yes | no | value on the next row | `Rounding Adj (.01` |
| test | 311 | total | `122.80` | 122.20 | yes | yes | total plus rounding | `Rounding Adj 0,02` |
| test | 315 | total | `11.25` | (none) | no | yes | - | `` |
| test | 316 | total | `1.38` | (none) | no | yes | - | `` |
| test | 342 | total | `19.31` | 19.34 | yes | no | last total line | `Total Payable: 19.34` |
| test | 343 | total | `51.88` | (none) | yes | no | - | `` |
| test | 381 | total | `RM111.90` | 10.56 | no | yes | last total line | `Total SV 210.56 Total GY : 10.56` |
| test | 387 | total | `82.80` | (none) | yes | no | - | `` |
| test | 395 | total | `RM8.35` | 0.47 | yes | no | value on the next row | `6h GST RMO.47` |
| test | 404 | total | `6.85` | (none) | yes | no | - | `` |
| test | 407 | total | `89.00` | 87.00 | yes | no | last total line | `Total RM 87.00` |
| test | 437 | total | `11.10` | 13.40 | yes | no | last total line | `Tatal Rounded 13.40 \|` |
| test | 508 | total | `16.00` | 6.06 | yes | yes | last total line | `TOTAL £6.06` |
| test | 544 | total | `11.60` | 11.66 | yes | no | last total line | `TOTAL 11.66` |
| test | 562 | total | `15.00` | 14.00 | yes | yes | last total line | `Tota] (Inclusive of GST): 14.00` |
| test | 575 | total | `137.15` | 137.45 | yes | no | last total line | `Total Sales (Inclusive 637) RM 137.45` |
| test | 584 | total | `5.00` | 10.00 | no | yes | last total line | `Accepted total 10.00 RH` |
| test | 610 | total | `85.20` | 65.20 | yes | yes | last total line | `Net Total 65.20` |
| test | 624 | total | `102.00` | 702.00 | yes | yes | last total line | `Total Amt Rounded ——S«702.00` |
