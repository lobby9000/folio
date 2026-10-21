# Folio

[![tests](https://github.com/lobby9000/folio/actions/workflows/tests.yml/badge.svg)](https://github.com/lobby9000/folio/actions/workflows/tests.yml)

Turns scanned documents into structured, searchable fields: OCR, field extraction with validation, document search, and an API served from AWS Lambda behind API Gateway. Built in stages, measuring each one before moving on.

[Versión en español](README.es.md)

## Status

| Stage | What it adds | Status |
|---|---|---|
| 1. OCR baseline | Tesseract, OpenCV preprocessing, CER, WER and field checks on SROIE receipts | Done |
| 2. Extraction baseline | Rules for the total and the date, normalised amounts and dates, a cash check | Done |
| 3. Learned and LLM extraction | A ranker learned from labelled receipts, and a cascade that asks an LLM reading the scan only when unsure | Done |
| 4. Classification and clustering | Document type from OCR text, and receipt layouts grouped without labels | Done |
| 5. Search | BM25, character n-grams and embeddings fused with RRF, scored with Recall@k and MRR in two languages | Done |
| 6. API and AWS | FastAPI in a Lambda container image behind API Gateway | Next |

<!-- results:start -->
## Results so far

### Stage 5: search over scanned pages

The 3,482 Tobacco3482 pages of stage 4 are cut into 5,561 chunks and searched with 200 questions, each asked in English and in Spanish. Half name something on the page (a person, a number, a code) and half only describe it. Three rankers and their fusions were compared on 50 dev questions, and the fusion chosen there was scored once on the other 150, alongside each ranker alone. Report: [search](results/search_eval.md).

| On the 150 test questions | English: Recall@10 [95% CI] | English: MRR@10 | Spanish: Recall@10 [95% CI] | Spanish: MRR@10 |
|---|---|---|---|---|
| BM25 over English word stems | 95% [91%, 98%] | 0.862 | 47% [40%, 55%] | 0.283 |
| TF-IDF of character n-grams | 91% [86%, 95%] | 0.800 | 67% [59%, 74%] | 0.529 |
| Embeddings (jina-embeddings-v2-base-es) | 81% [74%, 87%] | 0.697 | 79% [71%, 84%] | 0.608 |
| Character n-grams + embeddings, RRF with k = 30 (chosen on dev) | 94% [89%, 97%] | 0.820 | 84% [77%, 89%] | 0.697 |

Recall@10 is the share of questions whose page is among the first ten returned. MRR@10 averages 1/position of that page and counts zero past the tenth.

* **In English, BM25 is hard to beat.** It has the right page in its first ten results for 95% of the questions. The embeddings find no page that BM25 misses, while BM25 finds 21 that the embeddings miss, 19 of them for described questions (BM25 gets 91% of those, the embeddings 65%). A likely reason: every page in this archive is about the tobacco business, so the gist of a question matches many pages and its specific words decide. One caution: the questions were written while reading their pages and share 81% (exact) and 43% (described) of their content words with them, probably more than a user who has never seen the page would.
* **Across languages, the embeddings carry the search.** Asked in Spanish about English pages, BM25 keeps only the questions with a name or a number (93% of the exact ones, 1% of the described ones). Character n-grams rescue some through shared spellings (67%), and the bilingual embeddings reach 79%, with 50 pages that BM25 misses.
* **The fusion stays close to the best ranker in each language without knowing which language it gets.** Character n-grams with embeddings get 94% in English (BM25: 95%, exact McNemar p = 0.69) and 84% in Spanish (embeddings: 79%, p = 0.12), far above the weakest ranker in each: 19 more pages than the embeddings in English and 55 more than BM25 in Spanish (p < 0.001 both). The usual pairing, BM25 with embeddings, was a close second on dev (MRR@10 0.804 against 0.809) and gets 95% and 77% on test. Every fusion's test results are in the CSV.
* **What is left for the generation step.** Passing five pages to a model, the fusion has the right one for 89% of the English questions and 77% of the Spanish ones. With ten, 9 English and 24 Spanish questions still miss their page. A reranker over the first 20 pages, translating questions before BM25 and other embedding models are the next things to try.

### Stage 4: document types and receipt layouts

SROIE holds only receipts, so the document-type half of this stage uses another public set: the Tesseract text of Tobacco3482 (QS-OCR-Small), 3,482 scanned pages from one archive in ten classes. The layout half stays on the SROIE receipts and uses no labels to form its groups. Reports: [document types](results/classify_eval.md), [layouts](results/cluster_eval.md).

**Document type from OCR text**

| On the 696 held-out documents | Accuracy [95% CI] | Macro F1 |
|---|---|---|
| Always the most common class (memo) | 17.8% [15.2%, 20.8%] | 0.03 |
| Naive Bayes, words | 77.0% [73.7%, 80.0%] | 0.70 |
| Linear SVM, words | 85.2% [82.4%, 87.6%] | 0.84 |
| Linear SVM, characters inside words (chosen by cross-validation) | 84.8% [81.9%, 87.2%] | 0.84 |

* **A linear model on noisy OCR text gets 85% right.** Six candidates were compared by cross-validation on the 2,786 training documents, and character n-grams led there by about 2 points of macro F1. On the held-out documents they tie with words (590 against 593 right, exact McNemar p = 0.78), so the idea that characters cope better with OCR errors is not confirmed here. Naive Bayes trails both by 8 points.
* **The mistakes are between neighbours.** Emails (F1 0.98) and resumes (1.00) are almost never missed. The most common errors are letters read as memos (13 of 113), advertisements read as notes (9 of 46), and reports and scientific papers taken for each other. Five of those 13 letters have the word memorandum printed on them and most carry To, From and Subject lines, so some of these errors may be in the labels.
* **The margin is a usable confidence.** On the most confident half of the test documents the model is right 99.4% of the time. A threshold chosen on out-of-fold predictions of the training documents keeps 514 of the 696 at 96.5% and leaves 182 on which the model is right only 52% of the time: about a quarter of the documents, where a stronger model or a person should decide.
* **What the word model looks at.** Its heaviest words are "sent", "pm" and "subject" for emails, "dear" and "sincerely" for letters, "university" and "education" for resumes, "et al" and "receptor" for scientific papers, and "surgeon" and "warning" for advertisements, from the Surgeon General's warning printed on cigarette ads.

**Receipt layouts without labels**

* **The first rows of OCR text find the merchants.** Grouping the 200 test receipts by the characters of their first six rows gives 102 groups for 96 merchants: 94% of the pairs put together share a merchant, and 86% of the pairs that share one were put together (AMI 0.92). The representation and the cut were chosen on the dev and train receipts. Where the words sit on the page and a thumbnail of the scan did much worse there (AMI 0.58 and 0.48 against 0.91), so what a receipt says identifies its layout better than where its ink falls, at least with features this simple.
* **The cheap extractors do worse on new layouts. The LLM does not.** 142 of the 200 test receipts come from a merchant that also has dev or train receipts. On those the ranker is right 85% of the time, and on the other 58 it is right 74% of the time (rules: 80% and 64%). Haiku reading the scan gets 96% and 98%. The same split can be made without labels: a receipt counts as a known layout when its nearest dev or train receipt is closer than the clustering cut, which agrees with the merchant labels on 171 of 200 receipts, and splits the ranker into 85% and 71%.
* **The ranker did not memorise layouts.** Holding out whole merchants in cross-validation gives 80.5%, and whole groups found without labels 80.4%, against 80.6% for random receipts. The gap comes from the merchants themselves: with whole merchants held out, the ranker is right on 88% of receipts from merchants with five or more receipts and on 70% to 72% of rarer ones. Common chains print clean, regular receipts. A new layout is therefore a reason to ask the LLM, and the distance to the nearest known receipt flags one without labels.

### Stage 3: a learned ranker, then an LLM that reads the scan

Every amount OCR finds on a receipt becomes a candidate with 42 features, and a model learns from labelled receipts which candidate is the total. It was fitted on 426 receipts (the stage 2 development set plus 326 receipts no one had used), chosen by cross-validation there, and scored once on the same 200 held-out receipts as the rules. Then Claude Haiku 5.5 was asked for the total and the date through a JSON schema, once reading the OCR text and once reading the scanned image. The prompts and the routing were settled on the development receipts, and each way of asking ran once over all 200 test receipts, so the model alone and the cascade are scored on the same answers. Reports: [ranker](results/learned_eval.md), [Haiku reading the scan](results/llm_eval_test_all_image_anthropic_claude-haiku-5-5.md), [Haiku reading the OCR text](results/llm_eval_test_all_text_anthropic_claude-haiku-5-5.md).

| On the 200 held-out receipts | Total correct [95% CI] | Date correct [95% CI] | LLM cost per 1,000 receipts |
|---|---|---|---|
| Rules (stage 2), reading OCR text | 151, 76% [69%, 81%] | 161, 80% [74%, 85%] | none |
| Learned ranker, reading OCR text | 163, 82% [76%, 86%] | the rules' date | none |
| Haiku reading the OCR text | 157, 78% [72%, 84%] | 162, 81% [75%, 86%] | $0.13 |
| Cascade: rules or ranker when sure, otherwise Haiku reading the scan | 187, 94% [89%, 96%] | 190, 95% [91%, 97%] | $0.13 |
| Haiku reading the scan | 194, 97% [94%, 99%] | 198, 99% [96%, 100%] | $0.30 |

Costs are at list prices: $0.10 per million input tokens and $0.50 per million output tokens for Haiku 5.5.

**The ranker**

* **It beats the rules on the same receipts.** It is right on 15 receipts where the rules are wrong, against 3 the other way (exact McNemar p = 0.0075). Most of those 15 are receipts where no rule fired at all.
* **A simple model was enough.** Logistic regression and gradient-boosted trees tied in cross-validation (80.6% and 80.5% over 15 folds), so the logistic model was kept: its 42 weights fit in a JSON file and can be read. The heaviest say that the total tends to be the largest amount, the one the rules chose, a multiple of 5 sen (Malaysia rounds bills to 5 sen) and equal to cash minus change.
* **Its score is a usable confidence.** On the most confident half of the held-out receipts it is right 98 times out of 100.
* The rules, scored on the 326 receipts used here for training, find 73% of totals, close to their 76% on the test split. That is a second check that the test result was not luck.

**The LLM and the cascade**

* **Reading the scan beats reading the OCR text by 19 points.** Given the OCR rows, Haiku finds 78% of totals, fewer than the ranker. The right total is among the amounts OCR produced on only 176 of the 200 receipts (88%), and the answer check keeps only values printed in that text, so a total that OCR misread cannot be recovered. Given the image, the same model finds 97% of totals and 99% of dates. The development receipts had shown the same gap (79 against 96 of 99).
* **The cascade makes 89 calls for 200 receipts.** It keeps the rules' total when it matches cash minus change (71 receipts) and the ranker's when its score is at least 0.95 (53 receipts), a threshold chosen on the training receipts. It asks Haiku about the other 76, and about 13 of the trusted ones because the rules found no date. It finds 94% of totals, right on 25 receipts where the ranker is wrong and wrong on 1 where the ranker is right (exact McNemar p = 8e-07).
* **Asking about every receipt is still more accurate.** The model alone is right on 8 receipts where the cascade is wrong, against 1 the other way (p = 0.04), for $0.30 instead of $0.13 per 1,000 receipts. It also reads 8 dates right that the cascade takes from the rules and gets wrong. At these prices the cascade pays off when the number of calls is the limit, as under an API rate limit or when fewer receipts should leave the machine.
* **The trusted routes fail on digits, not on lines.** In all 8 of those receipts OCR had misread or dropped a digit in the lines the cascade relied on (102.00 read as 702.00, 104.00 as 404.00, a cash payment of 94.00 as 4.00). The ranker's score says how sure it is of the line, and a wrong digit on the right line looks just as sure.
* **The cash check has a blind spot.** When the rules find no total line, one of their fallbacks is cash minus change, and that answer then counts as confirmed by the same two lines. On the test receipts, a total read from a total line that matched cash minus change was right 62 times out of 62, and the fallback 7 times out of 9. The training receipts show the same gap (100 of 103 against 13 of 16), so the guard below trusts the check only for a total read from a total line.
* **Where the model reading the scan fails.** It misread a digit on 3 of the 200 receipts (one by a single cent) and took the wrong line on 2: an amount before tax from the tax summary, and an "accepted total" that was the cash handed over. On the sixth it read the printed grand total, 3.90, where the label records the 25.60 subtotal before a deduction. Of its 2 missed dates, 1 is the check's fault: the model read 25032018 correctly, and the check does not parse dates printed without separators.
* **Cost and speed.** A scan costs about 2,700 input tokens, and the median answer took 1.4 s with thinking turned off (Haiku 5.5 thinks by default). Every run of this stage, development receipts included, cost about $0.13. Answers are cached on disk under a hash of the prompt and the image, so regenerating a report sends no requests, and `--max-usd` stops a run before it spends more than a set amount.

**After stage 3: a guard against misread digits**

* **The design receipts fail the same way.** On the dev and train receipts, with out-of-fold ranker scores, the cascade trusts 270 totals and gets 11 wrong, all of them misread digits. OCR word confidence separates them badly: catching 8 of the 11 would send 75 right answers back to the LLM. Two other signals work better together: a second OCR pass with Otsu binarisation that does not read the same value, and a cash check that only confirms its own fallback. Together they catch 7 of the 11 and send back 32 right answers: 39 more calls for 425 receipts. Report: [`results/guard_design.md`](results/guard_design.md).
* **With the guard the cascade finds 96% of totals.** On the test receipts it is right on 191 of 200 with 103 calls, against 187 with 89 calls without the guard and 194 asking about every receipt ([report](results/llm_eval_test_all_image_anthropic_claude-haiku-5-5_guard.md)). This number is post hoc: the guard was designed after reading the test misses, so the test split is no longer blind for it. The answers came from the cache, so it cost nothing. The misses left are digits that both OCR passes read the same wrong way, and only a second look at the image fixes those.

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

## How stage 5 works

```
question --> BM25 over English word stems          --+
         --> TF-IDF of character n-grams           --+--> each ranks chunks, a page counts as its best chunk
         --> jina-embeddings-v2-base-es vectors    --+                    |
                                     Reciprocal Rank Fusion of the page rankings --> pages
```

* **Chunks** (`folio/search.py`): every page is cut at paragraph, line and word boundaries into chunks of up to 1,200 characters, as a RAG index would store them, and a last piece under 300 characters (a signature, a cc list) joins the chunk before it. The 3,482 pages give 5,561 chunks.
* **Rankers**: BM25 written here (k1 = 1.5, b = 0.75) over Snowball English stems without stop words; cosine similarity over TF-IDF of character 3 to 5 grams inside words, which still matches a word OCR damaged; and cosine similarity between vectors from jina-embeddings-v2-base-es, a Spanish and English model run on the CPU through fastembed. The vectors come from `scripts/embed_search.py`, which downloads the 640 MB model and stores each vector under the SHA-1 of its text, so a rerun embeds only what changed. The 5,561 chunks took about two hours on a laptop CPU, once.
* **Fusion**: Reciprocal Rank Fusion adds 1/(k + position) for every list a page appears in, so scores on different scales are never compared. Four combinations of the rankers, each with k = 10, 30 and 60, are compared on the dev questions by MRR@10. Character n-grams with embeddings at k = 30 won (0.809), just ahead of BM25 with embeddings at k = 10 (0.804).
* **Questions** (`benchmarks/search_questions.jsonl`): pages with at least 400 characters of OCR text were shuffled with a fixed seed and taken in order. For each page Claude (Anthropic's model) read the text and wrote one question under fixed rules, alternating two kinds: *exact* questions use an identifier from the page (a name, a number, a code), *described* ones use no names, numbers or codes. No question copies more than three consecutive words of its page beyond identifiers, which is checked automatically, and each has a Spanish translation. Pages that were unreadable, said nothing specific, or were templates that many pages repeat were skipped, and the file keeps every page looked at with the reason. The page answers its question. Among the three pages most similar to each source page (word TF-IDF), those at cosine 0.4 or more were read, and five of them, copies of the same document or equivalent forms, count as right answers too. The questions were split once, 50 for dev and 150 for test, half of each kind.
* **Evaluation** (`folio/evaluate_search.py`): Recall@1, 5, 10 and 20 and MRR@10 per language and kind of question, Wilson intervals, exact McNemar tests between the fusion and each ranker alone, and how many questions each ranker finds that the other misses.

## How stage 4 works

```
OCR text of a page -> character n-grams inside words (TF-IDF) -> linear SVM -> class + margin
      margin below the threshold?          -> a stronger model or a person decides

OCR of a receipt -> characters of its first six rows (TF-IDF) -> cosine distance -> average-linkage clusters
      nearest training receipt farther than the cut? -> a new layout
```

* **Document type** (`folio/classify.py`, `folio/evaluate_classify.py`): QS-OCR-Small is split once, stratified, into 2,786 training and 696 test documents. Six candidates (complement naive Bayes, and linear SVMs on word 1-2 grams, on character 2-5 grams inside words, or on both, at two regularisation strengths) are compared by five-fold cross-validation repeated twice, by macro F1, which weighs rare classes like common ones. The chosen one is fitted on all training documents and the test documents are scored once. The routing threshold on the margin is chosen on out-of-fold predictions. An LLM can be asked too (`--llm`), choosing from the classes through a JSON-schema enum.
* **Layouts** (`folio/cluster.py`, `folio/evaluate_cluster.py`): four representations (the OCR words, the characters of the first six rows, a grid of where the words sit, a thumbnail of the scan), compared by cosine distance and grouped by average-linkage agglomerative clustering cut at a distance threshold. Merchant names from the labels, with chain branches and label typos merged, only score the groups: adjusted mutual information, homogeneity, completeness, and pairwise precision and recall.
* **Grouped folds** (`folio/learn.py`): cross-validation can deal whole groups to folds, largest first, so a merchant or a layout found without labels never sits on both sides.

## How the learned stage works

```
amounts on the receipt -> 42 features each -> logistic regression -> best candidate + score
      rules confirmed by cash?  -> keep the rules' total
      score >= 0.95?            -> keep the ranker's total
      otherwise                 -> ask the LLM: the scan in, total and date out through a JSON schema
```

* **Candidates and features** (`folio/learn.py`): every amount on every row. Features describe the row (total, subtotal, tax, cash and change words, with the OCR misreadings the rules already accept), the rows around it, its place on the page, the height of its row, the OCR confidence, how often the same value repeats, whether it is a multiple of 5 sen, whether it equals cash minus change, and whether the rules picked it. Evidence is also pooled per value, because the same amount printed on the total line and on the cash line is one answer seen twice.
* **Model choice**: logistic regression and gradient-boosted trees, compared with 5-fold cross-validation repeated three times, always holding out whole receipts so a receipt never helps score itself. The routing threshold was chosen on out-of-fold predictions of the training receipts, never on the test split.
* **LLM extractor** (`folio/llm.py`): the model reads the scan or the OCR text. Reading the scan, it answers with each value and the whole line it read it from, and a value is kept if it parses as an amount or a date. Reading the text, it gets the OCR rows numbered and answers with each value and its row number, and a value printed nowhere in the text is rejected, so the model can choose but not invent. Both answer through structured outputs, with the JSON schema enforced by the API, on the Claude API or on a local model through Ollama. Answers are cached on disk under a hash of the prompt, the schema and the image.
* **Evaluation** (`folio/evaluate_learned.py`, `folio/evaluate_llm.py`): paired comparisons on the same receipts, Wilson intervals, accuracy by confidence, the cascade's routes, and the LLM's answers by status (read from the image, grounded in the text, cited the wrong row, not in the text).

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
pytest -q                                   # 187 tests in a few seconds

bash scripts/get_sroie.sh                   # 626 receipts into data/sroie (about 600 MB)
folio ocr data/sroie/data/img/000.jpg --prep gray --psm 6
folio ocr data/sroie/data/img/000.jpg --words          # each word with its box and confidence
folio eval-ocr --limit 100 --workers 2      # writes results/ocr_eval.md and results/ocr_eval.csv

folio extract data/sroie/data/img/097.jpg   # total and date, with the row each came from
folio extract data/sroie/data/img/097.jpg --json
folio eval-extract --workers 2              # OCR cached in cache/ocr, writes results/extract_eval.md and .csv

folio eval-learned                          # cross-validation, one test run, saves models/total_ranker.json
folio extract data/sroie/data/img/097.jpg --ranker models/total_ranker.json

bash scripts/get_tobacco_text.sh            # 3,482 OCR texts of Tobacco3482 into data/qs-ocr-small (2.4 MB)
folio eval-classify                         # document types: cross-validation, one test run, results/classify_eval.md
folio eval-cluster                          # receipt layouts without labels, results/cluster_eval.md

folio export-search                         # chunks and queries to embed, into data/search
pip install fastembed==0.9.0 && python scripts/embed_search.py data/search   # downloads 640 MB once, about two hours on a laptop CPU
folio eval-search                           # dev questions only: compares the fusions
folio eval-search --split all               # the one test run, results/search_eval.md

# The LLM stage. With a local model, reading the OCR text (nothing leaves the machine):
ollama pull qwen2.5:3b
folio eval-llm --backend ollama --model qwen2.5:3b          # asks only about the receipts the cascade routes
# Or with the Claude API (if the API asks for a workspace, also export ANTHROPIC_WORKSPACE_ID):
pip install anthropic && export ANTHROPIC_API_KEY=...
folio eval-llm --backend anthropic --input image --mode all   # the report above: Haiku reads all 200 test scans, about 6 cents
folio eval-llm --backend anthropic --input image --mode all --guard   # with the guard: answers come from the cache
folio extract data/sroie/data/img/097.jpg --ranker models/total_ranker.json --llm anthropic --llm-input image --guard
folio eval-classify --llm anthropic        # also ask Haiku about the documents the classifier is unsure of
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
  learn.py        candidates, features and the learned ranker
  evaluate_learned.py  model choice by cross-validation and the test run
  llm.py          LLM backends, the JSON schema, the evidence check and the cascade
  evaluate_llm.py  scoring of the LLM and the cascade
  classify.py     document types from OCR text: the candidate classifiers and the LLM prompt
  evaluate_classify.py  model choice by cross-validation, the test run and the report
  cluster.py      receipt representations, layout clustering and merchant keys
  evaluate_cluster.py  layout groups scored against merchants, known and new layouts, grouped folds
  search.py       chunks, BM25, character n-grams, embeddings and Reciprocal Rank Fusion
  evaluate_search.py  the questions, Recall@k and MRR, the fusion chosen on dev and the test run
  cli.py          folio ocr, extract and one eval-* command per experiment
models/           the fitted ranker as JSON
benchmarks/       the stage 5 questions, one line per page looked at
tests/            unit tests and end-to-end checks with drawn images, no dataset files
scripts/          dataset downloads, the guard's design data and the embedding script
results/          experiment outputs
```

## Data

[QS-OCR-Small](https://github.com/Quicksign/ocrized-text-dataset) (Quicksign, 2019) is the Tesseract 4 text of the Tobacco3482 dataset (Kumar et al., 2013): 3,482 scanned pages from the Truth Tobacco Industry Documents archive in ten classes (advertisement, email, form, letter, memo, news, note, report, resume and scientific paper). Stage 4 classifies document types on it, because SROIE holds only receipts, and stage 5 searches its pages. The download script pins the archive's checksum. The dataset has no licence of its own, so it is used for research and is not stored here. It is split once, stratified, into 2,786 training and 696 test documents.

[SROIE](https://rrc.cvc.uab.es/?ch=13) (ICDAR 2019) has scanned receipts from Malaysia with line-level text boxes and four key fields: company, date, address and total. The download script uses the [zzzDavid/ICDAR-2019-SROIE](https://github.com/zzzDavid/ICDAR-2019-SROIE) mirror. The data is for research use and is not stored in this repository. The 626 receipts are split once: 100 for development (the stage 1 sample), 200 held out for testing, and the other 326 for training. The 326 stayed unused through stage 2 and became the ranker's training data in stage 3, so later changes are judged by cross-validation on them and by the one test run each method gets. Its transcriptions have some noise of their own (a box file may say `SDN BND` where the key file says `SDN BHD`), so even perfect OCR would not reach zero error.

## Licence

Code under the [MIT licence](LICENSE).

---

Part of the portfolio of Luis Ángel Quiroz López · [lobby9000.github.io](https://lobby9000.github.io) · [GitHub](https://github.com/lobby9000)
