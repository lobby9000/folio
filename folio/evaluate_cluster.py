"""Score layout clustering on SROIE, and what layouts say about the extractors.

The representation and the distance threshold are chosen on the dev and train receipts
(merchant names score the candidates). The test receipts are then grouped once, alone,
and scored the same way. Two uses of the groups are measured: telling a known layout
from a new one by the distance to the nearest training receipt, and holding out whole
layouts in cross-validation to see whether the ranker of stage 3 was rewarded for
memorising them.
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .cluster import KINDS, cluster, distances, merchant, nearest_distance, represent, scores
from .evaluate_learned import label
from .extract import extract_total
from .learn import TotalRanker, cross_validate, group_folds, make_model, pick, stack
from .metrics import wilson_interval
from .normalize import parse_amount
from .ocr import OcrResult
from .sroie import Receipt

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class ClusterRow:
    receipt: str
    merchant: str
    group: int
    seen_merchant: bool  # the merchant has receipts among the dev and train receipts
    nearest: float  # cosine distance to the nearest dev or train receipt
    known_layout: bool  # nearest closer than the clustering threshold
    rules_ok: bool | None
    ranker_ok: bool | None
    llm_ok: bool | None  # Haiku reading the scan, from the stage 3 report


@dataclass
class Report:
    design_size: int
    design_merchants: int
    selection: list[tuple[str, float, dict]]
    kind: str
    threshold: float
    design_detection: dict[str, int]
    test_scores: dict
    rows: list[ClusterRow]
    largest: list[tuple[int, list[tuple[str, int]]]]
    folds: dict[str, list[float]]
    by_frequency: dict[str, tuple[int, int]]


def _detection(known: Sequence[bool], truth: Sequence[bool]) -> dict[str, int]:
    return {"known_right": sum(k and t for k, t in zip(known, truth, strict=True)),
            "known_wrong": sum(k and not t for k, t in zip(known, truth, strict=True)),
            "new_right": sum(not k and not t for k, t in zip(known, truth, strict=True)),
            "new_wrong": sum(not k and t for k, t in zip(known, truth, strict=True))}


def _out_of_fold_correct(receipts, groups, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Whether the ranker gets each receipt right when its whole group is held out."""
    fold = group_folds(groups, folds, seed)
    ok = np.zeros(len(receipts), dtype=bool)
    for f in range(folds):
        if not (fold == f).any():
            continue
        X, y, _ = stack([r for r, k in zip(receipts, fold, strict=True) if k != f])
        model = make_model("logistic").fit(X, y)
        for i in np.flatnonzero(fold == f):
            got = pick(model, receipts[i])
            ok[i] = got is not None and got[0].value == receipts[i].gold
    return ok


def run(design: Sequence[tuple[Receipt, OcrResult]], test: Sequence[tuple[Receipt, OcrResult]],
        ranker: TotalRanker | None = None, llm_ok: dict[str, bool] | None = None,
        thresholds: Sequence[float] = THRESHOLDS, repeats: int = 3,
        log: Callable[[str], None] = lambda _: None) -> Report:
    d_receipts, d_ocr = [r for r, _ in design], [o for _, o in design]
    t_receipts, t_ocr = [r for r, _ in test], [o for _, o in test]
    d_truth = [merchant(r.fields["company"]) for r in d_receipts]
    t_truth = [merchant(r.fields["company"]) for r in t_receipts]

    selection = []
    for kind in KINDS:
        log(f"grouping the dev and train receipts by {kind}")
        D = distances(represent(kind, d_receipts, d_ocr)[0])
        for t in thresholds:
            selection.append((kind, t, scores(d_truth, cluster(D, t))))
    kind, threshold, _ = max(selection, key=lambda s: s[2]["ami"])

    Xd, fitted = represent(kind, d_receipts, d_ocr)
    Dd = distances(Xd)
    counts = Counter(d_truth)
    design_detection = _detection(list(nearest_distance(Dd, exclude_self=True) < threshold),
                                  [counts[m] > 1 for m in d_truth])

    log("grouping the test receipts, alone")
    t_groups = cluster(distances(represent(kind, t_receipts, t_ocr)[0]), threshold)
    test_scores = scores(t_truth, t_groups)
    sizes = Counter(t_groups.tolist())
    largest = [(size, Counter(m for m, g in zip(t_truth, t_groups, strict=True) if g == group).most_common(3))
               for group, size in sizes.most_common(5)]

    Xt_known = represent(kind, t_receipts, t_ocr, fitted)[0]
    near = distances(Xt_known, Xd).min(axis=1)
    seen = set(d_truth)
    rows = []
    for k, (receipt, ocr) in enumerate(test):
        gold = parse_amount(receipt.fields["total"]) if receipt.fields["total"] else None
        rules = extract_total(ocr.text.splitlines())
        ranked = ranker.extract(ocr.rows) if ranker is not None else None
        rows.append(ClusterRow(
            receipt=receipt.id, merchant=t_truth[k], group=int(t_groups[k]), seen_merchant=t_truth[k] in seen,
            nearest=round(float(near[k]), 4), known_layout=bool(near[k] < threshold),
            rules_ok=None if gold is None else (rules is not None and rules.value == gold),
            ranker_ok=(None if gold is None or ranker is None
                       else ranked is not None and ranked.value == gold),
            llm_ok=None if llm_ok is None or receipt.id not in llm_ok else llm_ok[receipt.id],
        ))

    log("cross-validating the ranker with whole layouts held out")
    pairs = [(label(r, o), merchant(r.fields["company"])) for r, o in design if r.fields["total"]]
    labelled = [lr for lr, _ in pairs]
    merchants = [m for _, m in pairs]
    with_total = [k for k, (r, _) in enumerate(design) if r.fields["total"]]
    d_groups = cluster(Dd, threshold)
    found = [str(d_groups[k]) for k in with_total]
    def repeated(groups):
        return [s for seed in range(repeats) for s in cross_validate("logistic", labelled, 5, seed, groups)]

    folds = {"random receipts (stage 3)": repeated(None), "whole merchants": repeated(merchants),
             "whole groups found without labels": repeated(found)}
    ok = _out_of_fold_correct(labelled, merchants)
    freq = Counter(merchants)
    by_frequency = {}
    for name, lo, hi in (("1", 1, 1), ("2 to 4", 2, 4), ("5 or more", 5, 10**9)):
        idx = [i for i, m in enumerate(merchants) if lo <= freq[m] <= hi]
        by_frequency[name] = (int(ok[idx].sum()), len(idx))
    return Report(len(design), len(seen), selection, kind, threshold, design_detection, test_scores, rows,
                  largest, folds, by_frequency)


def write_csv(rows: Sequence[ClusterRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in fields(ClusterRow)])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _rate(k: int, n: int) -> str:
    if not n:
        return "n/a"
    lo, hi = wilson_interval(k, n)
    return f"{k} of {n} ({100 * k / n:.0f}%) [{100 * lo:.0f}%, {100 * hi:.0f}%]"


def render_markdown(report: Report, header: str = "") -> str:
    lines = ["# Receipt layouts without labels", ""]
    if header:
        lines += [header, ""]
    lines += ["## Choosing a representation on the dev and train receipts", "",
              f"{report.design_size} receipts from {report.design_merchants} merchants. Best threshold for "
              "each representation, by adjusted mutual information (AMI: 1 is a perfect match with the "
              "merchants, "
              "0 is chance).", "",
              "| Representation | Threshold | AMI | ARI | Groups | Homogeneity | Completeness |",
              "|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        k, t, s = max((x for x in report.selection if x[0] == kind), key=lambda x: x[2]["ami"])
        mark = " (chosen)" if (k, t) == (report.kind, report.threshold) else ""
        lines.append(f"| {k}{mark} | {t} | {s['ami']:.3f} | {s['ari']:.3f} | {s['groups']} | "
                     f"{s['homogeneity']:.3f} | {s['completeness']:.3f} |")

    s = report.test_scores
    lines += ["", "## Test: the 200 held-out receipts, grouped alone", "",
              f"`{report.kind}`, cut at cosine distance {report.threshold}, fitted on the test receipts "
              "only.", "",
              "| Measure | Value |", "|---|---|",
              f"| Groups found / merchants | {s['groups']} / {s['merchants']} |",
              f"| AMI / ARI | {s['ami']:.3f} / {s['ari']:.3f} |",
              f"| Homogeneity / completeness | {s['homogeneity']:.3f} / {s['completeness']:.3f} |",
              f"| Pairs put together that share a merchant | {100 * s['pair_precision']:.1f}% |",
              f"| Pairs sharing a merchant that were put together | {100 * s['pair_recall']:.1f}% |", "",
              "Largest groups: " + "; ".join(
                  f"{size} receipts ({', '.join(f'{m} {n}' for m, n in members)})"
                  for size, members in report.largest) + ".", ""]

    rows = report.rows
    seen = [r for r in rows if r.seen_merchant]
    new = [r for r in rows if not r.seen_merchant]
    d = report.design_detection
    known_ok = sum(r.known_layout == r.seen_merchant for r in rows)
    lines += ["## Known and new layouts", "",
              f"{len(seen)} of the {len(rows)} test receipts come from a merchant that also has dev or train "
              "receipts. A receipt counts as a known layout when its nearest dev or train receipt is closer "
              f"than {report.threshold}. On the dev and train receipts, leaving each one out, that rule "
              f"calls {d['known_right'] + d['known_wrong']} layouts known ({d['known_right']} rightly) and "
              f"{d['new_right'] + d['new_wrong']} new ({d['new_right']} rightly). On the test receipts it "
              f"agrees with the merchant labels on {known_ok} of {len(rows)}.", "",
              "| Extractor (total) | Merchant seen in dev or train | New merchant | Known layout, no labels "
              "| New layout, no labels |", "|---|---|---|---|---|"]
    known = [r for r in rows if r.known_layout]
    unknown = [r for r in rows if not r.known_layout]
    for name, attr in (("Rules (stage 2)", "rules_ok"), ("Ranker (stage 3)", "ranker_ok"),
                       ("Haiku reading the scan (stage 3)", "llm_ok")):
        cells = []
        for group in (seen, new, known, unknown):
            vals = [getattr(r, attr) for r in group if getattr(r, attr) is not None]
            cells.append(_rate(sum(vals), len(vals)) if vals else "n/a")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += ["", "## Honest folds for the ranker", "",
              "Five-fold cross-validation of the stage 3 ranker on the dev and train receipts, three "
              "shuffles, holding out random receipts or whole layouts.", "",
              "| Held out together | Accuracy, mean over folds |", "|---|---|"]
    for name, values in report.folds.items():
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        lines.append(f"| {name} | {100 * statistics.mean(values):.1f}% ± {100 * sd:.1f} |")
    lines += ["", "With whole merchants held out, accuracy by how many dev and train receipts the merchant "
              "has:", "", "| Receipts from the merchant | Ranker right |", "|---|---|"]
    for name, (k, n) in report.by_frequency.items():
        lines.append(f"| {name} | {_rate(k, n)} |")
    return "\n".join(lines) + "\n"
