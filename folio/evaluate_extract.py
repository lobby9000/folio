"""Score the field extractors on SROIE.

Every receipt is scored on two texts: the OCR output from stage 1's best setting,
and the dataset's own transcription ("reference"). The gap between the two says
how much of the error comes from reading and how much from the rules.

Values are compared after normalising (`5,90` equals `RM 5.90`, `19/10/18` equals
`19-10-2018`). For each miss on OCR text the evaluation asks two questions: do the
same rules get it right on the reference text (then OCR caused the miss), and did
the right value appear anywhere in the OCR text (if not, no extractor that picks
from the text could have found it).
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .evaluate import exact_mcnemar
from .extract import EXTRACTORS, Extraction
from .metrics import field_found, wilson_interval
from .normalize import find_amounts, find_dates, parse_amount, parse_date
from .sroie import Receipt

FIELDS = ("total", "date")
_PARSERS: dict[str, Callable] = {"total": parse_amount, "date": parse_date}


def _values_in(field: str, lines: Sequence[str]) -> set:
    if field == "total":
        return {a.value for line in lines for a in find_amounts(line)}
    return {d.value for line in lines for d in find_dates(line)}


@dataclass
class FieldRow:
    split: str
    receipt: str
    source: str  # "ocr" or "reference"
    method: str  # "rules" or "baseline"
    field: str
    gold: str
    predicted: str
    correct: bool | None  # None when the label is empty or cannot be read
    in_text: bool | None  # the right value appears somewhere in the text
    verbatim: bool | None  # the label appears character for character (stage 1's check)
    rule: str
    checked: bool | None
    evidence: str


def score_receipt(split: str, receipt: Receipt, source: str, lines: Sequence[str]) -> list[FieldRow]:
    rows = []
    text = "\n".join(lines)
    for field in FIELDS:
        gold = receipt.fields[field]
        gold_value = _PARSERS[field](gold) if gold else None
        present = _values_in(field, lines)
        for method, extractors in EXTRACTORS.items():
            got: Extraction | None = extractors[field](lines)
            rows.append(FieldRow(
                split=split, receipt=receipt.id, source=source, method=method, field=field, gold=gold,
                predicted="" if got is None else str(got.value),
                correct=None if gold_value is None else (got is not None and got.value == gold_value),
                in_text=None if gold_value is None else gold_value in present,
                verbatim=field_found(gold, text) if gold else None,
                rule="" if got is None else got.rule,
                checked=None if got is None else got.checked,
                evidence="" if got is None else got.evidence.strip(),
            ))
    return rows


@dataclass
class FieldSummary:
    split: str
    source: str
    method: str
    field: str
    n: int
    correct: int
    in_text: int  # the right value appears somewhere in the text
    verbatim: int  # the label appears character for character

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else float("nan")

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_interval(self.correct, self.n)


def summarize(rows: Sequence[FieldRow]) -> list[FieldSummary]:
    groups: dict[tuple[str, str, str, str], list[FieldRow]] = {}
    for r in rows:
        if r.correct is not None:
            groups.setdefault((r.split, r.source, r.method, r.field), []).append(r)
    out = []
    for (split, source, method, field), rs in groups.items():
        out.append(FieldSummary(
            split, source, method, field, n=len(rs),
            correct=sum(r.correct for r in rs),
            in_text=sum(bool(r.in_text) for r in rs),
            verbatim=sum(bool(r.verbatim) for r in rs),
        ))
    return out


@dataclass
class Paired:
    split: str
    source: str
    field: str
    n: int
    only_rules: int
    only_baseline: int

    @property
    def p_value(self) -> float:
        return exact_mcnemar(self.only_rules, self.only_baseline)


def compare_methods(rows: Sequence[FieldRow]) -> list[Paired]:
    by_key: dict[tuple, dict[str, bool]] = {}
    for r in rows:
        if r.correct is not None:
            by_key.setdefault((r.split, r.source, r.field, r.receipt), {})[r.method] = r.correct
    groups: dict[tuple[str, str, str], list[dict[str, bool]]] = {}
    for (split, source, field, _), result in by_key.items():
        if {"rules", "baseline"} <= result.keys():
            groups.setdefault((split, source, field), []).append(result)
    return [
        Paired(split, source, field, n=len(rs),
               only_rules=sum(r["rules"] and not r["baseline"] for r in rs),
               only_baseline=sum(r["baseline"] and not r["rules"] for r in rs))
        for (split, source, field), rs in groups.items()
    ]


@dataclass
class CheckSummary:
    split: str
    status: str  # "agrees", "disagrees" or "nothing to check"
    n: int
    correct: int


def summarize_checks(rows: Sequence[FieldRow], source: str = "ocr") -> list[CheckSummary]:
    """How often the total is right, split by whether it matched cash minus change."""
    names = {True: "agrees", False: "disagrees", None: "nothing to check"}
    out = []
    for split in dict.fromkeys(r.split for r in rows):
        rs = [r for r in rows if r.split == split and r.source == source and r.method == "rules"
              and r.field == "total" and r.correct is not None]
        for status in (True, False, None):
            group = [r for r in rs if r.checked is status]
            out.append(CheckSummary(split, names[status], len(group), sum(r.correct for r in group)))
    return out


def write_csv(rows: Sequence[FieldRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in fields(FieldRow)])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def read_csv(path: Path) -> list[FieldRow]:
    def opt(v: str) -> bool | None:
        return None if v == "" else v == "True"

    with Path(path).open(newline="", encoding="utf-8") as f:
        return [FieldRow(**{**r, "correct": opt(r["correct"]), "in_text": opt(r["in_text"]),
                            "verbatim": opt(r["verbatim"]), "checked": opt(r["checked"])})
                for r in csv.DictReader(f)]


def _rate(k: int, n: int) -> str:
    if not n:
        return "n/a"
    lo, hi = wilson_interval(k, n)
    return f"{k} of {n} ({100 * k / n:.0f}%) [{100 * lo:.0f}%, {100 * hi:.0f}%]"


_SOURCE = {"ocr": "OCR", "reference": "reference"}


def render_markdown(rows: Sequence[FieldRow], header: str = "") -> str:
    summaries = summarize(rows)
    get = {(s.split, s.source, s.method, s.field): s for s in summaries}
    splits = list(dict.fromkeys(r.split for r in rows))
    lines = ["# Field extraction: total and date", ""]
    if header:
        lines += [header, ""]
    lines += ["## Accuracy", "",
              "Correct answers out of the receipts that have a label, with 95% Wilson intervals.", "",
              "| Split | Text | Method | Total | Date |", "|---|---|---|---|---|"]
    for split in splits:
        for source in ("ocr", "reference"):
            for method in ("rules", "baseline"):
                t, d = get.get((split, source, method, "total")), get.get((split, source, method, "date"))
                if t is None and d is None:
                    continue
                cells = [_rate(s.correct, s.n) if s else "n/a" for s in (t, d)]
                lines.append(f"| {split} | {_SOURCE[source]} | {method} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Where the misses come from", "",
              "Rules on OCR text. A miss is put down to OCR when the same rules get that receipt right on "
              "the reference text. The last two columns are ceilings: how often the right value appears "
              "anywhere in the OCR text once normalised, and how often the label appears character for "
              "character (the check used in stage 1).", "",
              "| Split | Field | Correct | Missed | Missed because of OCR | Missed on the reference text too "
              "| Value anywhere in the text | Label verbatim in the text |",
              "|---|---|---|---|---|---|---|---|"]
    outcome = {(r.split, r.source, r.field, r.receipt): r.correct
               for r in rows if r.method == "rules" and r.correct is not None}
    for split in splits:
        for field in FIELDS:
            s = get.get((split, "ocr", "rules", field))
            if not s:
                continue
            missed = [k for k, ok in outcome.items() if k[:3] == (split, "ocr", field) and not ok]
            both = sum(outcome.get((split, "reference", field, k[3])) is False for k in missed)
            lines.append(f"| {split} | {field} | {s.correct} of {s.n} | {len(missed)} | {len(missed) - both} "
                         f"| {both} | {s.in_text} | {s.verbatim} |")
    lines += ["", "## Rules against the baseline", "",
              "Same receipts, paired. The baseline takes the largest amount and the first date. "
              "p is an exact McNemar test on the receipts where only one of the two is right.", "",
              "| Split | Text | Field | Only rules right | Only baseline right | p |",
              "|---|---|---|---|---|---|"]
    for p in compare_methods(rows):
        lines.append(f"| {p.split} | {_SOURCE[p.source]} | {p.field} | {p.only_rules} | {p.only_baseline} "
                     f"| {p.p_value:.2g} |")
    lines += ["", "## Checking the total against cash minus change", "",
              "Rules on OCR text, split by whether the extracted total equals cash tendered minus change.",
              "",
              "| Split | Check | Receipts | Total correct |", "|---|---|---|---|"]
    for c in summarize_checks(rows):
        lines.append(f"| {c.split} | {c.status} | {c.n} | {_rate(c.correct, c.n)} |")
    misses = [r for r in rows if r.source == "ocr" and r.method == "rules" and r.correct is False]
    if misses:
        lines += ["", "## Every miss of the rules on OCR text", "",
                  "| Split | Receipt | Field | Label | Extracted | Right on reference text "
                  "| Value in the OCR text | Rule | Line used |", "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(misses, key=lambda r: (splits.index(r.split), r.field, r.receipt)):
            evidence = r.evidence.replace("|", "\\|").replace("`", "'")[:70]
            on_ref = outcome.get((r.split, "reference", r.field, r.receipt))
            lines.append(f"| {r.split} | {r.receipt} | {r.field} | `{r.gold}` | {r.predicted or '(none)'} "
                         f"| {'yes' if on_ref else 'no'} | {'yes' if r.in_text else 'no'} | {r.rule or '-'} "
                         f"| `{evidence}` |")
    return "\n".join(lines) + "\n"
