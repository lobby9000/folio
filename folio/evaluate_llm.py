"""Score the LLM extractor and the cascade on SROIE receipts.

Two ways to run it:

* `cascade` (default): the model is asked only about receipts the cheap extractors
  are unsure of, as it would be in production. This measures what the cascade gains
  and what it costs.
* `all`: the model is asked about every receipt. This also measures the model on its
  own, and shows what the cascade would have decided with the same answers.
"""

from __future__ import annotations

import csv
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .evaluate import exact_mcnemar
from .evaluate_extract import parse_label_date
from .extract import extract_date, extract_total
from .learn import TotalRanker
from .llm import Answer, Routed, amounts_read, cascade, consult_image, consult_text
from .metrics import wilson_interval
from .normalize import parse_amount
from .ocr import OcrResult
from .sroie import Receipt


@dataclass
class LlmRow:
    receipt: str
    gold_total: str
    gold_date: str
    route: str
    rules_total_ok: bool
    ranker_total_ok: bool
    final_total: str
    final_total_ok: bool
    final_date: str
    final_date_ok: bool | None
    asked: bool
    llm_total: str
    llm_total_status: str
    llm_total_ok: bool | None
    llm_date_status: str
    llm_date_ok: bool | None
    seconds: float | None
    input_tokens: int | None
    output_tokens: int | None


def evaluate(
    receipts: Sequence[tuple[Receipt, OcrResult]],
    ranker: TotalRanker,
    ask: Callable[..., Answer],
    mode: str = "cascade",
    threshold: float = 0.95,
    source: str = "text",
    second: Mapping[str, OcrResult] | None = None,
) -> list[LlmRow]:
    """`ask(receipt_id, prompt, image=None, schema=None)` returns the model's answer, cached or
    not. `source` is "text" (the model reads the OCR rows) or "image" (it reads the scan).
    `second`, a second OCR pass per receipt, turns on the cascade's guard against misread digits."""
    rows = []
    for receipt, ocr in receipts:
        if not receipt.fields["total"]:
            continue
        lines = ocr.text.splitlines()
        gold_total = parse_amount(receipt.fields["total"])
        gold_date = parse_label_date(receipt.fields["date"]) if receipt.fields["date"] else None
        rules_total, rules_date = extract_total(lines), extract_date(lines)
        ranked = ranker.extract(ocr.rows)

        def consult(receipt: Receipt = receipt, lines: list[str] = lines):
            def ask_this(prompt, image=None, schema=None):
                return ask(receipt.id, prompt, image, schema)

            if source == "image":
                return consult_image(ask_this, Path(receipt.image_path).read_bytes(), lines)
            return consult_text(ask_this, lines)

        first = consult() if mode == "all" else None
        reading = None if second is None else amounts_read(second[receipt.id].text)
        routed: Routed = cascade(lines, rules_total, rules_date, ranked,
                                 (lambda first=first: first) if first is not None else consult, threshold,
                                 reading)
        answer, checked = first if first is not None else (routed.answer, routed.llm)
        t, d = checked.get("total"), checked.get("date")
        rows.append(LlmRow(
            receipt=receipt.id, gold_total=receipt.fields["total"], gold_date=receipt.fields["date"],
            route=routed.route,
            rules_total_ok=rules_total is not None and rules_total.value == gold_total,
            ranker_total_ok=ranked is not None and ranked.value == gold_total,
            final_total="" if routed.total is None else str(routed.total.value),
            final_total_ok=routed.total is not None and routed.total.value == gold_total,
            final_date="" if routed.date is None else str(routed.date.value),
            final_date_ok=None if gold_date is None else (
                routed.date is not None and routed.date.value == gold_date),
            asked=answer is not None,
            llm_total="" if t is None else t.claimed,
            llm_total_status="" if t is None else t.status,
            llm_total_ok=None if t is None else t.value == gold_total,
            llm_date_status="" if d is None else d.status,
            llm_date_ok=None if d is None or gold_date is None else d.value == gold_date,
            seconds=None if answer is None else round(answer.seconds, 3),
            input_tokens=None if answer is None else answer.input_tokens,
            output_tokens=None if answer is None else answer.output_tokens,
        ))
    return rows


def write_csv(rows: Sequence[LlmRow], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in fields(LlmRow)])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _rate(k: int, n: int) -> str:
    if not n:
        return "n/a"
    lo, hi = wilson_interval(k, n)
    return f"{k} of {n} ({100 * k / n:.0f}%) [{100 * lo:.0f}%, {100 * hi:.0f}%]"


def render_markdown(rows: Sequence[LlmRow], header: str = "", source: str = "text") -> str:
    n = len(rows)
    asked = [r for r in rows if r.asked]
    routed = [r for r in rows if r.route == "llm"]
    lines = ["# LLM extraction and the cascade", ""]
    if header:
        lines += [header, ""]
    lines += ["## Total", "", "| Extractor | Total correct [95% CI] |", "|---|---|",
              f"| Rules (stage 2) | {_rate(sum(r.rules_total_ok for r in rows), n)} |",
              f"| Learned ranker | {_rate(sum(r.ranker_total_ok for r in rows), n)} |",
              f"| Cascade: rules or ranker when sure, the LLM otherwise | "
              f"{_rate(sum(r.final_total_ok for r in rows), n)} |"]
    if len(asked) == n:
        right = sum(bool(r.llm_total_ok) for r in rows)
        name = "LLM alone, reading the scan" if source == "image" else "LLM alone (grounded answers only)"
        lines.append(f"| {name} | {_rate(right, n)} |")
    only_c = sum(r.final_total_ok and not r.ranker_total_ok for r in rows)
    only_r = sum(r.ranker_total_ok and not r.final_total_ok for r in rows)
    lines += ["", f"Cascade against the ranker, paired: only the cascade right on {only_c}, only the ranker "
              "right "
              f"on {only_r} (exact McNemar p = {exact_mcnemar(only_c, only_r):.2g}).", ""]

    lines += ["## Where each receipt went", "", "| Route | Receipts | Total correct |", "|---|---|---|"]
    routes = ("confirmed by cash", "confident ranker", "second reading disagrees", "llm", "llm for the date")
    for route in routes:
        group = [r for r in rows if r.route == route]
        if not group and route == "second reading disagrees":
            continue  # the guard was off
        right = sum(r.final_total_ok for r in group)
        lines.append(f"| {route} | {len(group)} | {_rate(right, len(group))} |")
    if routed:
        lines += ["", f"On the {len(routed)} receipts sent to the LLM, the ranker alone would have been "
              "right on "
                  f"{sum(r.ranker_total_ok for r in routed)}.", ""]

    rule = ("A value read from the scan is kept if it parses as an amount or a date." if source == "image"
            else "A total is kept only if its value is printed somewhere in the OCR text.")
    lines += ["## What the model answered", "", rule, "",
              "| Status of the model's total | Answers | Right value |", "|---|---|---|"]
    statuses = ("grounded", "wrong row", "not in the text", "read from the image", "unreadable", "no answer")
    for status in statuses:
        group = [r for r in asked if r.llm_total_status == status]
        if group:
            lines.append(f"| {status} | {len(group)} | {sum(bool(r.llm_total_ok) for r in group)} |")

    dated = [r for r in rows if r.final_date_ok is not None]
    date_ok = sum(bool(r.final_date_ok) for r in dated)
    lines += ["", "## Date", "", "| Extractor | Date correct [95% CI] |", "|---|---|",
              "| Cascade: the rules' date, or the model's when it was asked "
              f"| {_rate(date_ok, len(dated))} |"]
    llm_dated = [r for r in rows if r.llm_date_ok is not None]
    if len(asked) == n and llm_dated:
        lines.append(f"| LLM alone | {_rate(sum(bool(r.llm_date_ok) for r in llm_dated), len(llm_dated))} |")
    lines += ["", "The cascade also asks the model when the rules found no date.", ""]

    secs = [r.seconds for r in asked if r.seconds is not None]
    tin = [r.input_tokens for r in asked if r.input_tokens]
    tout = [r.output_tokens for r in asked if r.output_tokens]
    lines += ["## Cost", "", f"{len(asked)} of {n} receipts were sent to the model."]
    if secs:
        lines.append(f"Median time per answer {statistics.median(secs):.1f} s, slowest {max(secs):.1f} s.")
    if tin:
        lines.append(f"Tokens: {sum(tin):,} in and {sum(tout):,} out in total, "
                     f"{statistics.mean(tin):,.0f} in per receipt.")
    return "\n".join(lines) + "\n"
