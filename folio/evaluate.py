"""Compare preprocessing pipelines and Tesseract page modes on SROIE receipts.

Every configuration runs on the same receipts, so configurations can be compared
pair by pair on each receipt instead of only through their averages.
"""

from __future__ import annotations

import csv
import math
import os
import random
import statistics
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from .metrics import bag_of_words, cer, field_found, wer, wilson_interval
from .ocr import OcrConfig, run_ocr
from .preprocess import PIPELINES
from .sroie import Receipt


@dataclass(frozen=True)
class Row:
    """The result of one configuration on one receipt."""

    receipt: str
    pipeline: str
    psm: int
    cer: float
    wer: float
    word_f1: float
    total_found: bool | None  # None when the receipt has no labelled total
    date_found: bool | None
    mean_conf: float | None
    seconds: float

    @property
    def config(self) -> tuple[str, int]:
        return (self.pipeline, self.psm)


def _found(value: str, text: str) -> bool | None:
    """Whether a labelled field appears in the text, or None if the label is empty.

    Some SROIE key files leave a field blank. Those receipts are left out of that
    field's rate instead of counting as a miss for every configuration.
    """
    return field_found(value, text) if value.strip() else None


def _evaluate_receipt(job: tuple[Receipt, Sequence[str], Sequence[int], str]) -> list[Row]:
    receipt, pipelines, psms, lang = job
    # One Tesseract thread per process: with several worker processes this is faster
    # than letting each Tesseract call compete for every core.
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    img = cv2.imread(str(receipt.image_path))
    if img is None:
        raise FileNotFoundError(receipt.image_path)
    reference = receipt.reference_text
    rows = []
    for name in pipelines:
        start = time.perf_counter()
        prepared = PIPELINES[name](img)
        prep_seconds = time.perf_counter() - start
        for psm in psms:
            start = time.perf_counter()
            text = run_ocr(prepared, OcrConfig(lang=lang, psm=psm))
            seconds = prep_seconds + time.perf_counter() - start
            rows.append(Row(
                receipt=receipt.id,
                pipeline=name,
                psm=psm,
                cer=cer(reference, text.text),
                wer=wer(reference, text.text),
                word_f1=bag_of_words(reference, text.text).f1,
                total_found=_found(receipt.fields["total"], text.text),
                date_found=_found(receipt.fields["date"], text.text),
                mean_conf=text.mean_confidence,
                seconds=seconds,
            ))
    return rows


def evaluate(
    receipts: Iterable[Receipt],
    pipelines: Sequence[str],
    psms: Sequence[int],
    lang: str = "eng",
    workers: int = 1,
) -> list[Row]:
    jobs = [(r, tuple(pipelines), tuple(psms), lang) for r in receipts]
    if workers <= 1:
        return [row for job in jobs for row in _evaluate_receipt(job)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [row for rows in pool.map(_evaluate_receipt, jobs) for row in rows]


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean, resampling receipts."""
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class Summary:
    pipeline: str
    psm: int
    n: int
    cer_mean: float
    cer_ci: tuple[float, float]
    cer_median: float
    wer_mean: float
    word_f1_mean: float
    total_found: int
    total_n: int  # receipts with a labelled total
    date_found: int
    date_n: int
    seconds_mean: float

    @property
    def total_ci(self) -> tuple[float, float]:
        return wilson_interval(self.total_found, self.total_n)


def summarize(rows: Sequence[Row]) -> list[Summary]:
    """One summary per configuration, best mean CER first."""
    groups: dict[tuple[str, int], list[Row]] = {}
    for row in rows:
        groups.setdefault(row.config, []).append(row)
    summaries = []
    for (pipeline, psm), group in groups.items():
        cers = [r.cer for r in group]
        summaries.append(Summary(
            pipeline=pipeline,
            psm=psm,
            n=len(group),
            cer_mean=statistics.fmean(cers),
            cer_ci=bootstrap_ci(cers),
            cer_median=statistics.median(cers),
            wer_mean=statistics.fmean(r.wer for r in group),
            word_f1_mean=statistics.fmean(r.word_f1 for r in group),
            total_found=sum(r.total_found is True for r in group),
            total_n=sum(r.total_found is not None for r in group),
            date_found=sum(r.date_found is True for r in group),
            date_n=sum(r.date_found is not None for r in group),
            seconds_mean=statistics.fmean(r.seconds for r in group),
        ))
    return sorted(summaries, key=lambda s: s.cer_mean)


@dataclass(frozen=True)
class Comparison:
    """Configuration A against B on the same receipts."""

    a: tuple[str, int]
    b: tuple[str, int]
    n: int
    cer_diff_mean: float  # mean of CER(A) - CER(B); negative means A reads better
    cer_diff_ci: tuple[float, float]
    a_better: int  # receipts where A has the lower CER
    b_better: int
    total_only_a: int  # total found by A and missed by B
    total_only_b: int
    total_p: float  # exact McNemar test on the total being found


def compare(rows: Sequence[Row], a: tuple[str, int], b: tuple[str, int]) -> Comparison:
    by_receipt: dict[str, dict[tuple[str, int], Row]] = {}
    for row in rows:
        by_receipt.setdefault(row.receipt, {})[row.config] = row
    pairs = [(v[a], v[b]) for v in by_receipt.values() if a in v and b in v]
    diffs = [ra.cer - rb.cer for ra, rb in pairs]
    labelled = [(ra, rb) for ra, rb in pairs if ra.total_found is not None and rb.total_found is not None]
    only_a = sum(bool(ra.total_found) and not rb.total_found for ra, rb in labelled)
    only_b = sum(bool(rb.total_found) and not ra.total_found for ra, rb in labelled)
    return Comparison(
        a=a,
        b=b,
        n=len(pairs),
        cer_diff_mean=statistics.fmean(diffs),
        cer_diff_ci=bootstrap_ci(diffs),
        a_better=sum(d < 0 for d in diffs),
        b_better=sum(d > 0 for d in diffs),
        total_only_a=only_a,
        total_only_b=only_b,
        total_p=exact_mcnemar(only_a, only_b),
    )


def write_csv(rows: Sequence[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_markdown(
    summaries: Sequence[Summary], comparison: Comparison | None, header: str
) -> str:
    lines = [
        header,
        "",
        "| Preprocessing | PSM | Mean CER [95% CI] | Median CER | WER | Word F1 "
        "| Total found [95% CI] | Date found | Seconds per receipt |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lo, hi = s.cer_ci
        tlo, thi = s.total_ci
        lines.append(
            f"| {s.pipeline} | {s.psm} | {_pct(s.cer_mean)} [{_pct(lo)}, {_pct(hi)}] | {_pct(s.cer_median)} "
            f"| {_pct(s.wer_mean)} | {s.word_f1_mean:.3f} | {s.total_found}/{s.total_n} "
            f"({_pct(s.total_found / max(s.total_n, 1))}) [{_pct(tlo)}, {_pct(thi)}] "
            f"| {s.date_found}/{s.date_n} | {s.seconds_mean:.2f} |"
        )
    if comparison is not None:
        c = comparison
        lo, hi = c.cer_diff_ci
        lines += [
            "",
            f"**Best configuration against the default** ({c.a[0]}, PSM {c.a[1]} against {c.b[0]}, "
            f"PSM {c.b[1]}, on the same {c.n} receipts):",
            "",
            f"* Mean CER difference: {100 * c.cer_diff_mean:+.1f} points [95% CI {100 * lo:+.1f}, "
            f"{100 * hi:+.1f}]. Lower CER on {c.a_better} receipts, higher on {c.b_better}.",
            f"* Total found only by the first: {c.total_only_a}. Only by the second: {c.total_only_b}. "
            f"Exact McNemar p = {c.total_p:.3g}.",
        ]
    return "\n".join(lines) + "\n"
