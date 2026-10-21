# Search over scanned pages

Tesseract text of the Tobacco3482 pages (QS-OCR-Small, `data/qs-ocr-small`), questions from `benchmarks/search_questions.jsonl`. Embeddings: jina-embeddings-v2-base-es through fastembed (`data/search/embeddings.npz`, made by `scripts/embed_search.py`). BM25 uses k1 = 1.5, b = 0.75 over English Snowball stems; character n-grams are TF-IDF over 3 to 5 characters inside words.

3482 pages split into 5561 chunks of up to 1,200 characters; a page ranks by its best chunk. Questions: 200, written for pages drawn at random (234 looked at, 34 skipped: 13 template, 11 no content, 10 unreadable). Recall@k is the share of questions whose page (or a copy of it) is among the first k pages returned; MRR@10 averages 1/position of that page, counting 0 past the tenth.

## Choosing the fusion on the dev questions

MRR@10 over the 50 dev questions asked in English and in Spanish (100 queries). Reciprocal Rank Fusion adds 1/(k + position) from each list.

| Fused methods | k=10 | k=30 | k=60 |
|---|---|---|---|
| BM25 + embeddings | 0.804 | 0.760 | 0.748 |
| char + embeddings | 0.804 | **0.809** | 0.771 |
| BM25 + char | 0.636 | 0.633 | 0.632 |
| BM25 + char + embeddings | 0.755 | 0.753 | 0.741 |

Chosen: `char + embeddings (RRF, k=30)`.

## Dev: 50 questions

**Asked in English**

| Method | Recall@1 | Recall@5 | Recall@10 [95% CI] | Recall@20 | MRR@10 |
|---|---|---|---|---|---|
| BM25 | 86% | 94% | 94% [84%, 98%] | 94% | 0.891 |
| character n-grams | 76% | 88% | 92% [81%, 97%] | 92% | 0.815 |
| embeddings | 72% | 84% | 84% [71%, 92%] | 88% | 0.772 |
| char + embeddings (RRF, k=30) | 78% | 92% | 92% [81%, 97%] | 98% | 0.832 |

**Asked in Spanish**

| Method | Recall@1 | Recall@5 | Recall@10 [95% CI] | Recall@20 | MRR@10 |
|---|---|---|---|---|---|
| BM25 | 26% | 40% | 46% [33%, 60%] | 56% | 0.312 |
| character n-grams | 44% | 58% | 66% [52%, 78%] | 70% | 0.508 |
| embeddings | 72% | 80% | 84% [71%, 92%] | 90% | 0.759 |
| char + embeddings (RRF, k=30) | 74% | 84% | 92% [81%, 97%] | 92% | 0.786 |

## By kind of question

Recall@10 on the dev questions. Exact questions name something on the page; described ones only say what it is about.

| Method | exact, English | described, English | exact, Spanish | described, Spanish |
|---|---|---|---|---|
| BM25 | 100% | 88% | 92% | 0% |
| character n-grams | 100% | 84% | 96% | 36% |
| embeddings | 88% | 80% | 88% | 80% |
| char + embeddings (RRF, k=30) | 96% | 88% | 100% | 84% |

## What each method finds that the other misses

Dev questions whose page is in the first 10 results.

| Language | BM25 only | embeddings only | both | neither | `char + embeddings (RRF, k=30)` |
|---|---|---|---|---|---|
| English | 6 | 1 | 41 | 2 | 46 of 50 |
| Spanish | 2 | 21 | 21 | 6 | 46 of 50 |

Exact McNemar tests on Recall@10, `char + embeddings (RRF, k=30)` against each method alone:

* English: BM25: 1 gained, 2 lost, p = 1; character n-grams: 2 gained, 2 lost, p = 1; embeddings: 4 gained, 0 lost, p = 0.12.
* Spanish: BM25: 23 gained, 0 lost, p = 2.4e-07; character n-grams: 13 gained, 0 lost, p = 0.00024; embeddings: 4 gained, 0 lost, p = 0.12.

## How much the questions borrow from their pages

Share of a question's content words (stop words dropped, stemmed) that appear on its page, averaged over all 200 questions. Spanish questions share almost nothing but names and numbers.

| Kind | English | Spanish |
|---|---|---|
| exact | 81% | 30% |
| described | 43% | 4% |
