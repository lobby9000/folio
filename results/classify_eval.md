# Document type from OCR text

Tesseract text of the Tobacco3482 documents (QS-OCR-Small, `data/qs-ocr-small`): 3482 documents in 10 classes, split once with seed 0 into 2786 for training and 696 held out for testing, the same share of every class in each.

## Model choice: cross-validation on the training documents

2786 training documents, stratified folds. Chosen by mean macro F1, which weighs every class the same however rare it is.

| Model | Accuracy | Macro F1 |
|---|---|---|
| naive Bayes, words | 76.5% ± 0.7 | 69.7% ± 0.8 |
| linear SVM, words, C=0.5 | 82.4% ± 0.8 | 79.3% ± 1.3 |
| linear SVM, words, C=2 | 83.1% ± 0.9 | 80.4% ± 1.2 |
| linear SVM, characters, C=0.5 | 84.1% ± 0.9 | 81.7% ± 1.2 |
| linear SVM, characters, C=2 (chosen) | 84.8% ± 0.7 | 82.6% ± 0.7 |
| linear SVM, words and characters, C=0.5 | 84.5% ± 0.6 | 81.9% ± 1.2 |

## Test: held-out documents, scored once

696 documents, the same share of every class.

| Model | Accuracy [95% CI] | Macro F1 |
|---|---|---|
| Baseline: always `Memo`, the most common class | 124 of 696 (17.8%) [15.2%, 20.8%] | 0.030 |
| naive Bayes, words | 536 of 696 (77.0%) [73.7%, 80.0%] | 0.698 |
| linear SVM, words, C=0.5 | 583 of 696 (83.8%) [80.8%, 86.3%] | 0.818 |
| linear SVM, words, C=2 | 593 of 696 (85.2%) [82.4%, 87.6%] | 0.839 |
| linear SVM, characters, C=0.5 | 593 of 696 (85.2%) [82.4%, 87.6%] | 0.841 |
| linear SVM, characters, C=2 (chosen) | 590 of 696 (84.8%) [81.9%, 87.2%] | 0.835 |
| linear SVM, words and characters, C=0.5 | 588 of 696 (84.5%) [81.6%, 87.0%] | 0.828 |

Chosen against naive Bayes, words, paired: only the chosen right on 82, only the other on 28 (exact McNemar p = 2.5e-07).

Chosen against linear SVM, words, C=0.5, paired: only the chosen right on 30, only the other on 23 (exact McNemar p = 0.41).

Chosen against linear SVM, words, C=2, paired: only the chosen right on 25, only the other on 28 (exact McNemar p = 0.78).

Chosen against linear SVM, characters, C=0.5, paired: only the chosen right on 7, only the other on 10 (exact McNemar p = 0.63).

Chosen against linear SVM, words and characters, C=0.5, paired: only the chosen right on 13, only the other on 11 (exact McNemar p = 0.84).

## By class, chosen model

| Class | Precision | Recall | F1 | Test documents |
|---|---|---|---|---|
| ADVE | 1.00 | 0.67 | 0.81 | 46 |
| Email | 0.98 | 0.97 | 0.98 | 120 |
| Form | 0.80 | 0.86 | 0.83 | 86 |
| Letter | 0.82 | 0.82 | 0.82 | 113 |
| Memo | 0.85 | 0.93 | 0.88 | 124 |
| News | 0.90 | 0.95 | 0.92 | 38 |
| Note | 0.67 | 0.78 | 0.72 | 40 |
| Report | 0.67 | 0.64 | 0.65 | 53 |
| Resume | 1.00 | 1.00 | 1.00 | 24 |
| Scientific | 0.81 | 0.67 | 0.74 | 52 |

Confusion matrix (rows: true class, columns: predicted):

| | ADVE | Email | Form | Letter | Memo | News | Note | Report | Resume | Scientific |
|---|---|---|---|---|---|---|---|---|---|---|
| **ADVE** | 31 | 1 | 0 | 0 | 1 | 2 | 9 | 2 | 0 | 0 |
| **Email** | 0 | 117 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Form** | 0 | 1 | 74 | 3 | 1 | 0 | 3 | 3 | 0 | 1 |
| **Letter** | 0 | 0 | 1 | 93 | 13 | 1 | 0 | 4 | 0 | 1 |
| **Memo** | 0 | 0 | 0 | 6 | 115 | 0 | 1 | 1 | 0 | 1 |
| **News** | 0 | 0 | 1 | 0 | 0 | 36 | 0 | 1 | 0 | 0 |
| **Note** | 0 | 0 | 5 | 1 | 2 | 0 | 31 | 0 | 0 | 1 |
| **Report** | 0 | 0 | 4 | 7 | 4 | 0 | 0 | 34 | 0 | 4 |
| **Resume** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 | 0 |
| **Scientific** | 0 | 0 | 6 | 2 | 0 | 1 | 2 | 6 | 0 | 35 |

Most common confusions: Letter read as Memo (13), ADVE read as Note (9), Report read as Letter (7), Scientific read as Report (6).

10 test documents have fewer than 20 characters of OCR text; the model is right on 8 of them.

## Confidence

| Most confident share of the test documents | Accuracy |
|---|---|
| 50% | 346 of 348 (99.4%) [97.9%, 99.8%] |
| 70% | 472 of 487 (96.9%) [95.0%, 98.1%] |
| 80% | 525 of 557 (94.3%) [92.0%, 95.9%] |
| 90% | 567 of 626 (90.6%) [88.0%, 92.6%] |
| 100% | 590 of 696 (84.8%) [81.9%, 87.2%] |

Routing threshold, chosen on out-of-fold predictions of the training documents so that the documents kept are right 95% of the time: margin 0.465. It keeps 75% of the training documents. On the test documents it keeps 514 (496 of 514 (96.5%) [94.5%, 97.8%] right) and would send 182 to a stronger model (94 of 182 (51.6%) [44.4%, 58.8%] right without one).

## What the word model looks at

The heaviest word n-grams of `linear SVM, words, C=2` for each class, fitted on the training documents.

| Class | Words that push towards it |
|---|---|
| ADVE | camels, warning, surgeon, 16e, een, cl |
| Email | sent, pm, subject, pm to, message, subject re |
| Form | date, form, fax, name, pages, number |
| Letter | dear, sincerely, mr, interoffice, letter, interoffice memorandum |
| Memo | memorandum, correspondence, office correspondence, inter, inter office, office |
| News | said, sua, ing, said the, nation, hed |
| Note | your information, the desk, desk, desk of, hl, for september |
| Report | tobacco, current, and, level, of nicotine, units |
| Resume | university, university of, biology, present, education, fellow |
| Scientific | al, period covered, et al, receptor, covered, specific |
