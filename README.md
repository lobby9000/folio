# Folio

[![tests](https://github.com/lobby9000/folio/actions/workflows/tests.yml/badge.svg)](https://github.com/lobby9000/folio/actions/workflows/tests.yml)

Turns scanned documents into structured, searchable fields: OCR, field extraction with validation, document search, and an API served from AWS Lambda behind API Gateway. Built in stages, measuring each one before moving on.

[Versión en español](README.es.md)

## Status

| Stage | What it adds | Status |
|---|---|---|
| 1. OCR baseline | Tesseract, OpenCV preprocessing, CER, WER and field checks on SROIE receipts | Done |
| 2. Extraction baseline | Rules for the total and the date, normalised amounts and dates, a cash check | Done |
| 3. Learned and LLM extraction | A line classifier, and an LLM with a JSON schema and textual evidence | Next |
| 4. Classification and clustering | Document type, and unlabelled grouping of layouts | Planned |
| 5. Search | BM25 and embeddings fused with RRF, scored with Recall@k and MRR | Planned |
| 6. API and AWS | FastAPI in a Lambda container image behind API Gateway | Planned |

<!-- results:start -->
## Results so far

### Stage 2: extracting the total and the date

The rules were written while looking only at 100 development receipts, then scored once on 200 held-out receipts. Answers are compared as values, so `RM 5.90` and `5,90` count as the same total. Full report, with every miss: [`results/extract_eval.md`](results/extract_eval.md).

| Receipts | Text the rules read | Total correct [95% CI] | Date correct [95% CI] |
|---|---|---|---|
| 200 held out | OCR (stage 1's best setting) | 151 of 200, 76% [69%, 81%] | 161 of 200, 80% [74%, 85%] |
| 200 held out | the dataset's own transcription | 190 of 200, 95% [91%, 97%] | 198 of 200, 99% [96%, 100%] |
| 100 development | OCR | 86 of 99, 87% [79%, 92%] | 84 of 100, 84% [76%, 90%] |
| 200 held out | OCR, baseline (largest amount, first date) | 77 of 200, 38% [32%, 45%] | 161 of 200, 80% [74%, 85%] |

* **Reading is now the bottleneck, not the rules.** On the dataset's transcription the same rules find 95% of totals and 99% of dates. On OCR text they find 76% and 80%. Of the 49 totals missed on OCR text, 39 are receipts the rules get right on clean text.
* **The held-out receipts cost 11 points.** The rules scored 87% on the receipts they were written on and 76% on new ones. The clearest case is the fallback that reads a value from the row below a total label: right on 3 of 3 development receipts and on 0 of 4 held-out ones, where it mostly picked up rounding and tax lines. This is why the test split was scored only once, and why 326 receipts are still unused.
* **Cash minus change works as a confidence check.** On 71 held-out receipts the extracted total equals cash tendered minus change, and it is right on 69 of them (97%). On the other 129 it is right 64% of the time. Stage 3 can send only those to a stronger extractor or to a person.
* **The labels disagree about rounding.** Most SROIE labels record the rounded amount paid, but some record the total before rounding (12.37 on a receipt that rounds to 12.35). No rule satisfies both, which caps the score on clean text.
* **For the date, the first valid date is as good as any rule.** Preferring a date next to a "Date" label never changed the answer. Of the 39 dates missed on OCR text, 37 are misread digits or unreadable lines. The other two fail on clean text too: a date glued to the next word (`21/05/2018TIME`) and one printed without separators.

### Stage 1: OCR baseline

100 SROIE receipts, the same ones for every configuration. Full table: [`results/ocr_eval.md`](results/ocr_eval.md).

| Preprocessing | Page mode (PSM) | Mean CER [95% CI] | Word F1 | Total found | Seconds per receipt |
|---|---|---|---|---|---|
| gray | 6, one block of text | 13.2% [11.3%, 15.6%] | 0.724 | 90 of 99 | 0.80 |
| gray | 3, Tesseract's default | 17.3% [14.1%, 21.0%] | 0.706 | 79 of 99 | 0.85 |
| Otsu | 6 | 15.7% [13.5%, 18.4%] | 0.664 | 79 of 99 | 0.71 |
| adaptive | 6 | 17.3% [15.2%, 19.6%] | 0.652 | 82 of 99 | 0.78 |
| 2x upscale | 6 | 16.3% [13.4%, 19.9%] | 0.696 | 80 of 99 | 1.86 |

* **The page mode mattered more than any image filter.** Reading each receipt as one block of text (PSM 6) cut mean CER by 4.1 points against Tesseract's default (95% CI 1.9 to 6.5), and found the total on 12 receipts where the default missed it, against 1 the other way (exact McNemar p = 0.003).
* **Binarising before Tesseract hurt.** Otsu raised the CER on 68 of the 100 receipts. A likely reason: Tesseract already thresholds internally, and a second global threshold can erase faint strokes of thermal print.
* **Upscaling 2x hurt and was more than twice as slow.** Deskewing barely changed anything, which suggests these receipts were scanned straight.
<!-- results:end -->

## How the extraction stage works

```
OCR words -> rows -> amounts and dates as values -> total rules, date rules -> value + row it came from + rule + cash check
```

* **Normalising** (`folio/normalize.py`): amounts become `Decimal` and dates become `datetime.date`. It accepts the quirks seen in the development receipts: a currency mark glued to the number (`RM9.75`, `<RM_34.80`), a comma as decimal separator, a stray space around the separator (`85. 54`), a dropped trailing zero (`174.9`), a missing leading zero (`.02`) and the letter O read for a zero. Dates are read day first, the convention in Malaysia, and implausible years are rejected.
* **Total** (`folio/extract.py`): a total line is one with the word total, or an OCR misreading one letter away, or words such as payable, nett, grand or "after rounding". Lookalike lines are skipped: subtotals, item counts, amounts before or without tax, the tax itself, savings and the tax summary table. Reading stops at the first payment line, and the last total line wins, because receipts print the pre-rounding total first. If that total is not a multiple of 5 sen and a rounding line follows, the rounded amount is used. Fallbacks, in order: a value on the row below a total label, cash minus change, and a card payment.
* **The cash check**: when a receipt prints cash tendered and change, their difference is a second reading of the total. Every answer reports whether it agrees, and when the two disagree and the difference is printed in the totals block, the value read twice wins.
* **Date**: the first valid date after a date label, otherwise the first valid date, skipping expiry and due dates.
* **Evaluation** (`folio/evaluate_extract.py`): every receipt is scored on its OCR text and on the dataset's transcription, against a naive baseline, with Wilson intervals and paired McNemar tests. Each OCR miss is classed by whether the rules get it right on clean text, and the report lists every miss with the row the rules used.
* **OCR cache** (`folio/cache.py`): each receipt's words, boxes and confidences are stored as JSON once, so rerunning the extraction takes seconds instead of minutes.

## How the OCR stage works

```
scan.jpg -> preprocessing (OpenCV) -> Tesseract (words, boxes, confidence) -> rows -> text
                                                                                    |
                         ground-truth lines -> same rows rule -> reference text ----+--> CER, WER, word F1, fields found
```

* **Preprocessing** (`folio/preprocess.py`): grayscale, median denoising, Otsu or adaptive binarisation, 2x upscaling and deskewing by projection profile. Each step is a pure function, and `PIPELINES` names the combinations the experiment compares.
* **OCR** (`folio/ocr.py`): Tesseract's LSTM engine through `image_to_data`, keeping every word's box and confidence for later steps.
* **Reading order** (`folio/layout.py`): SROIE annotates lines and Tesseract returns words, sometimes ordered differently. Both go through the same rule (rows from top to bottom, each read left to right), so the error rate measures reading and not layout disagreements.
* **Metrics** (`folio/metrics.py`): character and word error rates from edit distance, bag-of-words F1, and whether the receipt's total and date appear verbatim. Texts are compared case-insensitively because the SROIE transcriptions are mostly uppercase.
* **Statistics** (`folio/evaluate.py`): every configuration runs on the same receipts. Means come with 95% bootstrap intervals, proportions with Wilson intervals, and the best configuration is compared with Tesseract's defaults receipt by receipt, with an exact McNemar test on finding the total.

## Run it

Ubuntu or any Linux with Python 3.11 or newer:

```bash
sudo apt install tesseract-ocr tesseract-ocr-spa
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e . --no-deps
pytest -q                                   # 116 tests in a few seconds

bash scripts/get_sroie.sh                   # 626 receipts into data/sroie (about 600 MB)
folio ocr data/sroie/data/img/000.jpg --prep gray --psm 6
folio ocr data/sroie/data/img/000.jpg --words          # each word with its box and confidence
folio eval-ocr --limit 100 --workers 2      # writes results/ocr_eval.md and results/ocr_eval.csv

folio extract data/sroie/data/img/097.jpg   # total and date, with the row each came from
folio extract data/sroie/data/img/097.jpg --json
folio eval-extract --workers 2              # OCR cached in cache/ocr, writes results/extract_eval.md and .csv
```

## Layout

```
folio/
  preprocess.py   image steps and the named pipelines
  ocr.py          Tesseract wrapper that keeps boxes and confidences
  layout.py       boxes and the shared reading-order rule
  metrics.py      CER, WER, bag-of-words F1, field checks, Wilson intervals
  sroie.py        loader for the SROIE folders and the dev and test splits
  evaluate.py     OCR experiment runner, summaries and paired comparisons
  cache.py        OCR output stored per receipt as JSON
  normalize.py    amounts and dates read as values
  extract.py      rules for the total and the date, and the baselines
  evaluate_extract.py  scoring of the extractors and the report
  cli.py          folio ocr, extract, eval-ocr and eval-extract
tests/            unit tests and end-to-end checks with drawn images, no dataset files
scripts/          dataset download
results/          experiment outputs
```

## Data

[SROIE](https://rrc.cvc.uab.es/?ch=13) (ICDAR 2019) has scanned receipts from Malaysia with line-level text boxes and four key fields: company, date, address and total. The download script uses the [zzzDavid/ICDAR-2019-SROIE](https://github.com/zzzDavid/ICDAR-2019-SROIE) mirror. The data is for research use and is not stored in this repository. The 626 receipts are split once: 100 for development (the stage 1 sample), 200 held out for testing, and 326 kept unused for a fresh test after any change to the rules. Its transcriptions have some noise of their own (a box file may say `SDN BND` where the key file says `SDN BHD`), so even perfect OCR would not reach zero error.

## Licence

Code under the [MIT licence](LICENSE).

---

Part of the portfolio of Luis Ángel Quiroz López · [lobby9000.github.io](https://lobby9000.github.io) · [GitHub](https://github.com/lobby9000)
