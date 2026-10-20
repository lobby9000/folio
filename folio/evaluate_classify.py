"""Score document-type classifiers on QS-OCR-Small (the Tesseract text of Tobacco3482).

The protocol is the one used for the receipts: candidates are compared by repeated,
stratified cross-validation on the training documents only, the chosen one is fitted
on all of them, and the held-out test documents are scored once. Every candidate's
test score is reported, but only cross-validation decided which one is chosen.
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .classify import (
    CLASSES,
    KINDS,
    SCHEMA,
    SYSTEM,
    Doc,
    build_prompt,
    make_classifier,
    parse_kind,
    predict,
    top_terms,
)
from .evaluate import exact_mcnemar
from .llm import Answer
from .metrics import wilson_interval

TERMS_KIND = "linear SVM, words, C=2"  # the readable model whose weights the report shows
EMPTY_CHARS = 20  # fewer characters than this: OCR found next to nothing on the page


@dataclass
class DocRow:
    doc: str
    label: str
    chars: int
    predicted: str
    correct: bool
    margin: float
    routed: bool = False  # the margin is below the threshold: the cascade would ask the LLM
    llm: str = ""
    llm_correct: bool | None = None
    final: str = ""
    final_correct: bool | None = None
    seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class Report:
    n_train: int
    n_test: int
    cv: dict[str, list[tuple[float, float]]]
    chosen: str
    majority: str
    test: dict[str, list[str]]  # every candidate's predictions on the test documents
    rows: list[DocRow]  # the chosen model, document by document
    threshold: float
    kept_share_train: float
    terms: dict[str, list[str]] = field(default_factory=dict)


def _accuracy_f1(gold: Sequence[str], pred: Sequence[str]) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return float(accuracy_score(gold, pred)), float(f1_score(gold, pred, average="macro", zero_division=0))


def cross_validate(kind: str, docs: Sequence[Doc], folds: int = 5,
                   repeats: int = 2) -> list[tuple[float, float]]:
    """Accuracy and macro F1 of every fold, `repeats` times with different shuffles."""
    from sklearn.model_selection import StratifiedKFold

    texts = np.array([d.text for d in docs], dtype=object)
    labels = np.array([d.label for d in docs])
    out = []
    for seed in range(repeats):
        for fit, held in StratifiedKFold(folds, shuffle=True, random_state=seed).split(texts, labels):
            model = make_classifier(kind).fit(texts[fit], labels[fit])
            out.append(_accuracy_f1(labels[held], predict(model, texts[held])[0]))
    return out


def out_of_fold(kind: str, docs: Sequence[Doc], folds: int = 5,
                seed: int = 0) -> tuple[list[str], np.ndarray]:
    """Predictions and margins for every training document from a model that did not see it."""
    from sklearn.model_selection import StratifiedKFold

    texts = np.array([d.text for d in docs], dtype=object)
    labels = np.array([d.label for d in docs])
    pred = np.empty(len(docs), dtype=object)
    margin = np.zeros(len(docs))
    for fit, held in StratifiedKFold(folds, shuffle=True, random_state=seed).split(texts, labels):
        model = make_classifier(kind).fit(texts[fit], labels[fit])
        p, m = predict(model, texts[held])
        pred[held], margin[held] = p, m
    return [str(p) for p in pred], margin


def routing_threshold(correct: Sequence[bool], margins: np.ndarray, target: float = 0.95) -> float:
    """The lowest margin at which the documents kept (margin at or above it) are still right
    `target` of the time. Documents below it would go to a stronger model."""
    order = np.argsort(-np.asarray(margins), kind="stable")
    right = np.cumsum(np.asarray(correct, dtype=float)[order])
    share_right = right / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(share_right >= target)
    if not len(ok):
        return float("inf")
    return float(np.asarray(margins)[order][ok[-1]])


def run(train: Sequence[Doc], test: Sequence[Doc], kinds: Sequence[str] = tuple(KINDS), folds: int = 5,
        repeats: int = 2, target: float = 0.95, log: Callable[[str], None] = lambda _: None) -> Report:
    cv = {}
    for kind in kinds:
        log(f"cross-validating {kind}")
        cv[kind] = cross_validate(kind, train, folds, repeats)
    chosen = max(kinds, key=lambda k: statistics.mean(f1 for _, f1 in cv[k]))

    log(f"choosing the routing threshold for {chosen}")
    oof_pred, oof_margin = out_of_fold(chosen, train, folds)
    oof_right = [p == d.label for p, d in zip(oof_pred, train, strict=True)]
    threshold = routing_threshold(oof_right, oof_margin, target)
    kept_share = float(np.mean(oof_margin >= threshold))

    texts_train, labels_train = [d.text for d in train], [d.label for d in train]
    texts_test = [d.text for d in test]
    predictions, rows = {}, []
    for kind in kinds:
        log(f"fitting {kind} on all training documents and scoring the test documents")
        model = make_classifier(kind).fit(texts_train, labels_train)
        pred, margin = predict(model, texts_test)
        predictions[kind] = pred
        if kind == chosen:
            rows = [DocRow(d.id, d.label, len(d.text.strip()), p, p == d.label, round(float(m), 4),
                           bool(m < threshold), final=p, final_correct=p == d.label)
                    for d, p, m in zip(test, pred, margin, strict=True)]
    words = make_classifier(TERMS_KIND).fit(texts_train, labels_train)
    majority = Counter(labels_train).most_common(1)[0][0]
    return Report(len(train), len(test), cv, chosen, majority, predictions, rows, threshold, kept_share,
                  top_terms(words))


def ask_llm(rows: Sequence[DocRow], docs: Sequence[Doc], ask: Callable[..., Answer],
            mode: str = "cascade") -> None:
    """Fill in the LLM's answers: about every document (`all`) or only the routed ones (`cascade`).
    `ask(key, prompt, image, schema, system)` is a CachedBackend's `ask`. The cascade's final
    answer is the LLM's for routed documents, the classifier's for the rest."""
    by_id = {d.id: d for d in docs}
    for row in rows:
        if mode != "all" and not row.routed:
            continue
        answer = ask(row.doc.replace("/", "__"), build_prompt(by_id[row.doc].text), None, SCHEMA, SYSTEM)
        kind = parse_kind(answer.data)
        row.llm = kind or ""
        row.llm_correct = kind == row.label
        row.seconds, row.input_tokens, row.output_tokens = (round(answer.seconds, 3), answer.input_tokens,
                                                            answer.output_tokens)
        if row.routed and kind is not None:
            row.final = kind
        row.final_correct = row.final == row.label


def write_csv(rows: Sequence[DocRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in fields(DocRow)])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _rate(k: int, n: int) -> str:
    if not n:
        return "n/a"
    lo, hi = wilson_interval(k, n)
    return f"{k} of {n} ({100 * k / n:.1f}%) [{100 * lo:.1f}%, {100 * hi:.1f}%]"


def _mean_sd(values: Sequence[float]) -> str:
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{100 * statistics.mean(values):.1f}% ± {100 * sd:.1f}"


def render_markdown(report: Report, header: str = "", cost: str = "") -> str:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    rows = report.rows
    gold = [r.label for r in rows]
    labels = list(CLASSES)
    lines = ["# Document type from OCR text", ""]
    if header:
        lines += [header, ""]

    lines += ["## Model choice: cross-validation on the training documents", "",
              f"{report.n_train} training documents, stratified folds. Chosen by mean macro F1, which "
              "weighs every class the same however rare it is.", "",
              "| Model | Accuracy | Macro F1 |", "|---|---|---|"]
    for kind, results in report.cv.items():
        mark = " (chosen)" if kind == report.chosen else ""
        accuracy, macro_f1 = _mean_sd([a for a, _ in results]), _mean_sd([f for _, f in results])
        lines.append(f"| {kind}{mark} | {accuracy} | {macro_f1} |")

    lines += ["", "## Test: held-out documents, scored once", "",
              f"{report.n_test} documents, the same share of every class.", "",
              "| Model | Accuracy [95% CI] | Macro F1 |", "|---|---|---|"]
    majority = [report.majority] * len(gold)
    lines.append(f"| Baseline: always `{report.majority}`, the most common class | "
                 f"{_rate(sum(p == g for p, g in zip(majority, gold, strict=True)), len(gold))} | "
                 f"{_accuracy_f1(gold, majority)[1]:.3f} |")
    for kind, pred in report.test.items():
        mark = " (chosen)" if kind == report.chosen else ""
        right = sum(p == g for p, g in zip(pred, gold, strict=True))
        lines.append(f"| {kind}{mark} | {_rate(right, len(gold))} | {_accuracy_f1(gold, pred)[1]:.3f} |")
    chosen = report.test[report.chosen]
    for kind, pred in report.test.items():
        if kind == report.chosen:
            continue
        only_c = sum(c == g != p for c, p, g in zip(chosen, pred, gold, strict=True))
        only_o = sum(p == g != c for c, p, g in zip(chosen, pred, gold, strict=True))
        lines.append("")
        lines.append(f"Chosen against {kind}, paired: only the chosen right on {only_c}, only the other on "
                     f"{only_o} (exact McNemar p = {exact_mcnemar(only_c, only_o):.2g}).")
    lines.append("")

    precision, recall, f1, support = precision_recall_fscore_support(gold, chosen, labels=labels,
                                                                     zero_division=0)
    lines += ["## By class, chosen model", "", "| Class | Precision | Recall | F1 | Test documents |",
              "|---|---|---|---|---|"]
    for k, label in enumerate(labels):
        lines.append(f"| {label} | {precision[k]:.2f} | {recall[k]:.2f} | {f1[k]:.2f} | {support[k]} |")
    matrix = confusion_matrix(gold, chosen, labels=labels)
    lines += ["", "Confusion matrix (rows: true class, columns: predicted):", "",
              "| | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for k, label in enumerate(labels):
        lines.append(f"| **{label}** | " + " | ".join(str(v) for v in matrix[k]) + " |")
    off = [(int(matrix[i, j]), labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels))
           if i != j and matrix[i, j]]
    top = ", ".join(f"{t} read as {p} ({n})" for n, t, p in sorted(off, reverse=True)[:4])
    lines += ["", f"Most common confusions: {top}.", ""]

    empty = [r for r in rows if r.chars < EMPTY_CHARS]
    if empty:
        lines += [f"{len(empty)} test documents have fewer than {EMPTY_CHARS} characters of OCR text; the "
                  f"model is right on {sum(r.correct for r in empty)} of them.", ""]

    margins = np.array([r.margin for r in rows])
    correct = np.array([r.correct for r in rows])
    order = np.argsort(-margins, kind="stable")
    lines += ["## Confidence", "", "| Most confident share of the test documents | Accuracy |", "|---|---|"]
    for share in (0.5, 0.7, 0.8, 0.9, 1.0):
        n = max(1, int(round(share * len(rows))))
        lines.append(f"| {int(share * 100)}% | {_rate(int(correct[order[:n]].sum()), n)} |")
    kept = [r for r in rows if not r.routed]
    routed = [r for r in rows if r.routed]
    kept_right = _rate(sum(r.correct for r in kept), len(kept))
    routed_right = _rate(sum(r.correct for r in routed), len(routed))
    lines += ["", "Routing threshold, chosen on out-of-fold predictions of the training documents so that "
              f"the documents kept are right 95% of the time: margin {report.threshold:.3f}. It keeps "
              f"{100 * report.kept_share_train:.0f}% of the training documents. On the test documents it "
              f"keeps {len(kept)} ({kept_right} right) and would send {len(routed)} to a stronger model "
              f"({routed_right} right without one).", ""]

    asked = [r for r in rows if r.llm_correct is not None]
    if asked:
        lines += ["## The LLM", ""]
        if len(asked) == len(rows):
            llm_pred = [r.llm or "none" for r in rows]
            alone = _rate(sum(bool(r.llm_correct) for r in rows), len(rows))
            lines.append(f"LLM alone, every test document: {alone}, "
                         f"macro F1 {_accuracy_f1(gold, llm_pred)[1]:.3f}.")
        final = [r.final for r in rows]
        lines.append(f"Cascade, the classifier when sure and the LLM otherwise: "
                     f"{_rate(sum(bool(r.final_correct) for r in rows), len(rows))}, macro F1 "
                     f"{_accuracy_f1(gold, final)[1]:.3f}, with {len(routed)} calls.")
        llm_routed = [r for r in routed if r.llm_correct is not None]
        if llm_routed:
            lines.append(f"On the routed documents the LLM is right on "
                         f"{_rate(sum(bool(r.llm_correct) for r in llm_routed), len(llm_routed))}.")
        secs = [r.seconds for r in asked if r.seconds is not None]
        tin = [r.input_tokens for r in asked if r.input_tokens]
        if secs and tin:
            lines.append(f"Median time per answer {statistics.median(secs):.1f} s, "
                         f"{statistics.mean(tin):,.0f} input tokens per document.")
        if cost:
            lines.append(cost)
        lines.append("")

    lines += ["## What the word model looks at", "",
              f"The heaviest word n-grams of `{TERMS_KIND}` for each class, fitted on the training "
              "documents.", "",
              "| Class | Words that push towards it |", "|---|---|"]
    for label in labels:
        lines.append(f"| {label} | {', '.join(report.terms.get(label, []))} |")
    return "\n".join(lines) + "\n"
