# Folio

[![tests](https://github.com/lobby9000/folio/actions/workflows/tests.yml/badge.svg)](https://github.com/lobby9000/folio/actions/workflows/tests.yml)

Turns scanned documents into structured, searchable fields: OCR, field extraction with validation, document search, and an API served from AWS Lambda behind API Gateway. Built in stages, measuring each one before moving on.

[Versión en español](README.es.md)

## Status

| Stage | What it adds | Status |
|---|---|---|
| 1. OCR baseline | Tesseract, OpenCV preprocessing, CER, WER and field checks on SROIE receipts | Done |
| 2. Extraction baseline | Regex rules for date and total, scored per field | Next |
| 3. Learned and LLM extraction | A line classifier, and an LLM with a JSON schema and textual evidence | Planned |
| 4. Classification and clustering | Document type, and unlabelled grouping of layouts | Planned |
| 5. Search | BM25 and embeddings fused with RRF, scored with Recall@k and MRR | Planned |
| 6. API and AWS | FastAPI in a Lambda container image behind API Gateway | Planned |

<!-- results:start -->
## Results so far: OCR baseline

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
* **What the best setting still misses.** Of its 9 missed totals, 4 look recoverable by normalising numbers (a space inside the amount, a dropped trailing zero, an `RM_` prefix), 2 are character confusions (S for 5, O for 0, % for 7) and in 3 the total was not read in a usable form. One more receipt has no labelled total and is left out. That sets stage 2: an extractor that normalises amounts before comparing them.
<!-- results:end -->

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
pytest -q                                   # 37 tests in a few seconds

bash scripts/get_sroie.sh                   # 626 receipts into data/sroie (about 600 MB)
folio ocr data/sroie/data/img/000.jpg --prep gray --psm 6
folio ocr data/sroie/data/img/000.jpg --words          # each word with its box and confidence
folio eval-ocr --limit 100 --workers 2      # writes results/ocr_eval.md and results/ocr_eval.csv
```

## Layout

```
folio/
  preprocess.py   image steps and the named pipelines
  ocr.py          Tesseract wrapper that keeps boxes and confidences
  layout.py       boxes and the shared reading-order rule
  metrics.py      CER, WER, bag-of-words F1, field checks, Wilson intervals
  sroie.py        loader for the SROIE folders
  evaluate.py     experiment runner, summaries and paired comparisons
  cli.py          folio ocr and folio eval-ocr
tests/            unit tests and end-to-end checks with drawn images, no dataset files
scripts/          dataset download
results/          experiment outputs
```

## Data

[SROIE](https://rrc.cvc.uab.es/?ch=13) (ICDAR 2019) has scanned receipts from Malaysia with line-level text boxes and four key fields: company, date, address and total. The download script uses the [zzzDavid/ICDAR-2019-SROIE](https://github.com/zzzDavid/ICDAR-2019-SROIE) mirror. The data is for research use and is not stored in this repository. Its transcriptions have some noise of their own (a box file may say `SDN BND` where the key file says `SDN BHD`), so even perfect OCR would not reach zero error.

## Licence

Code under the [MIT licence](LICENSE).

---

Part of the portfolio of Luis Ángel Quiroz López · [lobby9000.github.io](https://lobby9000.github.io) · [GitHub](https://github.com/lobby9000)
