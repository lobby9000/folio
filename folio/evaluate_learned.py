"""Stage 3 experiment: a ranker learned from labelled receipts against the stage 2 rules.

Protocol, in this order:

1. Fit pool = the `dev` and `train` splits (426 receipts). The rules were written on
   `dev` only, so the 326 `train` receipts also give the rules a fresh score.
2. Both model kinds are compared with 5-fold cross-validation on the fit pool,
   repeated with three different shuffles and always holding out whole receipts.
   The better one (logistic regression on a tie) is refitted on the whole pool.
3. That model is scored once on the 200 `test` receipts, paired against the rules
   on the same receipts.

Only OCR text is used here: it is what the extractor sees in production.
"""

from __future__ import annotations

import csv
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from decimal import Decimal
from pathlib import Path

from .evaluate import exact_mcnemar
from .extract import extract_total
from .learn import FEATURES, LabelledReceipt, TotalRanker, candidates, cross_validate, pick
from .metrics import wilson_interval
from .normalize import parse_amount
from .ocr import OcrResult
from .sroie import Receipt

KINDS = ("logistic", "boosting")


def label(receipt: Receipt, ocr: OcrResult) -> LabelledReceipt:
    found, X = candidates(ocr.rows)
    gold = parse_amount(receipt.fields["total"]) if receipt.fields["total"] else None
    return LabelledReceipt(receipt.id, found, X, gold)


@dataclass
class ScoredReceipt:
    receipt: str
    gold: str
    rules: str
    rules_correct: bool
    rules_checked: bool | None
    ranker: str
    ranker_correct: bool
    ranker_score: float
    reachable: bool
    evidence: str


@dataclass
class Report:
    pool_size: int
    cv: dict[str, list[float]]
    chosen: str
    rules_on_train: tuple[int, int]
    rows: list[ScoredReceipt]
    weights: dict[str, float]

    def counts(self) -> dict[str, int]:
        r = self.rows
        return {
            "n": len(r),
            "rules": sum(x.rules_correct for x in r),
            "ranker": sum(x.ranker_correct for x in r),
            "reachable": sum(x.reachable for x in r),
            "only_ranker": sum(x.ranker_correct and not x.rules_correct for x in r),
            "only_rules": sum(x.rules_correct and not x.ranker_correct for x in r),
        }


def run(fit_pool: Sequence[tuple[Receipt, OcrResult, str]], test: Sequence[tuple[Receipt, OcrResult]],
        folds: int = 5, repeats: int = 3) -> tuple[Report, TotalRanker]:
    pool = [(label(r, o), split) for r, o, split in fit_pool if r.fields["total"]]
    labelled = [lr for lr, _ in pool]
    cv = {kind: [score for seed in range(repeats) for score in cross_validate(kind, labelled, folds, seed)]
          for kind in KINDS}
    chosen = max(KINDS, key=lambda k: (statistics.mean(cv[k]), k == "logistic"))
    ranker = TotalRanker.fit(labelled, chosen)

    train_only = [(r, o) for r, o, split in fit_pool if split == "train" and r.fields["total"]]
    rules_hits = sum(_rules_value(o) == parse_amount(r.fields["total"]) for r, o in train_only)

    rows = []
    for receipt, ocr in test:
        if not receipt.fields["total"]:
            continue
        lr = label(receipt, ocr)
        rules = extract_total(ocr.text.splitlines())
        got = pick(ranker.model, lr)
        rows.append(ScoredReceipt(
            receipt=receipt.id, gold=receipt.fields["total"],
            rules="" if rules is None else str(rules.value),
            rules_correct=rules is not None and rules.value == lr.gold,
            rules_checked=None if rules is None else rules.checked,
            ranker="" if got is None else str(got[0].value),
            ranker_correct=got is not None and got[0].value == lr.gold,
            ranker_score=0.0 if got is None else round(got[1], 4),
            reachable=lr.reachable,
            evidence="" if got is None else got[0].text.strip(),
        ))
    weights = TotalRanker.fit(labelled, "logistic").weights()
    return Report(len(labelled), cv, chosen, (rules_hits, len(train_only)), rows, weights), ranker


def _rules_value(ocr: OcrResult) -> Decimal | None:
    got = extract_total(ocr.text.splitlines())
    return None if got is None else got.value


def coverage_table(rows: Sequence[ScoredReceipt], shares: Sequence[float] = (0.5, 0.7, 0.8, 0.9, 1.0)):
    """Accuracy on the most confident share of receipts, by the ranker's score."""
    ranked = sorted(rows, key=lambda r: -r.ranker_score)
    out = []
    for share in shares:
        k = round(share * len(ranked))
        kept = ranked[:k]
        out.append((share, k, sum(r.ranker_correct for r in kept), kept[-1].ranker_score if kept else None))
    return out


ROUTE_THRESHOLD = 0.95  # chosen on out-of-fold predictions over the fit pool, never on test


def route_summary(rows: Sequence[ScoredReceipt], threshold: float = ROUTE_THRESHOLD) -> dict[str, int]:
    """Receipts the cascade would settle without an LLM: the rules' total confirmed by cash
    minus change, or the ranker at least `threshold` sure. The rest would go to the LLM."""
    confirmed = [r for r in rows if r.rules_checked]
    confident = [r for r in rows if not r.rules_checked and r.ranker_score >= threshold]
    rest = [r for r in rows if not r.rules_checked and r.ranker_score < threshold]
    return {
        "confirmed": len(confirmed), "confirmed_ok": sum(r.rules_correct for r in confirmed),
        "confident": len(confident), "confident_ok": sum(r.ranker_correct for r in confident),
        "rest": len(rest), "rest_ok": sum(r.ranker_correct for r in rest),
    }


def write_csv(rows: Sequence[ScoredReceipt], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in fields(ScoredReceipt)])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _rate(k: int, n: int) -> str:
    if not n:
        return "n/a"
    lo, hi = wilson_interval(k, n)
    return f"{k} of {n} ({100 * k / n:.0f}%) [{100 * lo:.0f}%, {100 * hi:.0f}%]"


def render_markdown(report: Report, header: str = "") -> str:
    c = report.counts()
    n = c["n"]
    lines = ["# Learned extraction of the total", ""]
    if header:
        lines += [header, ""]
    lines += ["## Model choice, by cross-validation on the fit pool", "",
              f"5 folds over {report.pool_size} labelled receipts, repeated with 3 shuffles, whole receipts "
              "held out. Accuracy = share of held-out receipts whose top-scoring amount is the labelled "
              "total.", "",
              "| Model | Mean accuracy | Range over the 15 folds |", "|---|---|---|"]
    for kind, scores in report.cv.items():
        mark = " (chosen)" if kind == report.chosen else ""
        lines.append(f"| {kind}{mark} | {100 * statistics.mean(scores):.1f}% "
                     f"| {100 * min(scores):.0f}% to {100 * max(scores):.0f}% |")
    hits, total = report.rules_on_train
    lines += ["", f"For reference, the stage 2 rules find the total on {_rate(hits, total)} of the `train` "
              "receipts, which nobody had looked at when the rules were written.", ""]
    lines += ["## Held-out test, scored once", "",
              "| Extractor | Total correct [95% CI] |", "|---|---|",
              f"| Rules (stage 2) | {_rate(c['rules'], n)} |",
              f"| Learned ranker ({report.chosen}) | {_rate(c['ranker'], n)} |",
              f"| Ceiling: the total is among the amounts OCR produced | {_rate(c['reachable'], n)} |", "",
              f"Paired on the same receipts: only the ranker right on {c['only_ranker']}, only the rules "
              f"right on {c['only_rules']} "
              f"(exact McNemar p = {exact_mcnemar(c['only_ranker'], c['only_rules']):.2g}).", ""]
    lines += ["## Confidence", "",
              "Test receipts sorted by the ranker's score. Keeping only the most confident ones shows how "
              "much a confidence threshold can trade coverage for accuracy.", "",
              "| Most confident share | Receipts | Correct | Lowest score kept |", "|---|---|---|---|"]
    for share, k, hits, low in coverage_table(report.rows):
        lines.append(f"| {share:.0%} | {k} | {_rate(hits, k)} | {low:.2f} |")
    rs = route_summary(report.rows)
    lines += ["", "## Routing for stage 3", "",
              "The cascade keeps the rules' total when it matches cash minus change, keeps the ranker's "
              "when "
              f"its score is at least {ROUTE_THRESHOLD} (a threshold chosen on the fit pool), and sends the "
              "rest to an LLM.", "",
              "| Route | Test receipts | Total correct without the LLM |", "|---|---|---|",
              f"| Confirmed by cash minus change | {rs['confirmed']} "
              f"| {_rate(rs['confirmed_ok'], rs['confirmed'])} |",
              f"| Confident ranker | {rs['confident']} | {_rate(rs['confident_ok'], rs['confident'])} |",
              f"| To the LLM | {rs['rest']} | {_rate(rs['rest_ok'], rs['rest'])} |"]
    lines += ["", "## What the logistic model weighs most", "",
              "Standardised weights: the change in log-odds of being the total for one standard deviation of "
              "the feature. Fitted on the whole pool.", "",
              "| Feature | Weight |", "|---|---|"]
    for name, w in list(report.weights.items())[:12]:
        lines.append(f"| `{name}` | {w:+.2f} |")
    misses = [r for r in report.rows if not r.ranker_correct]
    lines += ["", "## Every miss of the ranker on the test receipts", "",
              "| Receipt | Label | Ranker | Score | Rules | Reachable | Row used |",
              "|---|---|---|---|---|---|---|"]
    for r in misses:
        ev = r.evidence.replace("|", "\\|").replace("`", "'")[:60]
        lines.append(f"| {r.receipt} | `{r.gold}` | {r.ranker or '(none)'} | {r.ranker_score:.2f} "
                     f"| {r.rules or '(none)'}{' (right)' if r.rules_correct else ''} "
                     f"| {'yes' if r.reachable else 'no'} | `{ev}` |")
    assert len(FEATURES) == len(report.weights)
    return "\n".join(lines) + "\n"
