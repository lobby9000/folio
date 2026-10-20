"""Pick a receipt's total and date out of its text with hand-written rules.

This is the baseline that the learned and LLM extractors of stage 3 have to beat,
so every rule is simple enough to explain in one sentence:

Total
  1. A *total line* has the word total (or an OCR misreading one letter away),
     or says payable, nett, grand, rounded, "after rounding" or "amount due".
  2. Lines that only look like the total are excluded: subtotals, quantities,
     amounts before or without tax, tax itself, savings, discounts, partial
     supplies, item headers and payment lines.
  3. Reading stops at the first payment line (cash, change, card...) that follows
     a total with an amount, or at a tax summary table, whose "Total" row repeats
     other numbers.
  4. The last total line with an amount wins: receipts print the pre-rounding
     total first and the amount to pay last.
  5. Malaysia rounds bills to the nearest 5 sen. If the chosen total is not a
     multiple of 0.05 and a rounding line follows, the rounded value is used.
  6. When the receipt prints cash tendered and change, cash minus change is a
     second reading of the total. If it disagrees with the chosen value but matches
     another amount printed in the totals block, the value read twice wins.
  7. Fallbacks, in order: a total label whose amount slipped to the next row, cash
     minus change, and the amount of a card payment (cards pay the exact total).

Each answer carries the line it came from and the rule that produced it, and the
total is checked against cash minus change when the receipt prints both.

Date: the first valid date that follows a date label on its line, otherwise the
first valid date on the receipt, skipping expiry and due dates.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from rapidfuzz.distance import Levenshtein

from .normalize import Amount, find_amounts, find_dates


@dataclass(frozen=True)
class Extraction:
    value: Decimal | dt.date
    line: int  # index of the row the value was read from
    evidence: str  # that row, as read
    rule: str
    checked: bool | None = None  # total equals cash minus change; None if the receipt has no such pair
    score: float | None = None  # a learned extractor's confidence; the rules leave it empty


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.casefold())


def _is_total_word(word: str) -> bool:
    if "total" in word or word == "ttl":
        return True
    return 4 <= len(word) <= 6 and Levenshtein.distance(word, "total") <= 1


_NOT_TOTAL = re.compile(
    r"sub\W?t|tender|\bexc|before|saving|discount|suppl|\btax\b"
    r"|price|desc|change|\bcash\b|\bpaid\b|receiv|include[sd]|point|deposit"
)
_COUNTS = ("qty", "quantit", "item", "count", "pcs", "unit")
_GST = re.compile(r"gst|gs7|&st")
_INCL = re.compile(r"\binc(?:l|\b|\.|\])")
_PAYMENT = re.compile(r"\bcash\b|tender|\bpaid\b|payment|change|receiv|\bcard\b|\bvisa\b|master|credit|debit")
_CHANGE = re.compile(r"change|\bchg\b")
_CASH = re.compile(r"\bcash\b|tender|\bpaid\b|receiv")
_CARD = re.compile(r"\bcard\b|\bvisa\b|master|debit|credit")


def is_total_line(line: str) -> bool:
    low = line.casefold()
    words = _words(low)
    labelled = (
        any(_is_total_word(w) for w in words)
        or bool({"payable", "nett", "grand", "rounded"} & set(words))
        or ("after" in words and any(w.startswith(("round", "adj")) for w in words))
        or ("due" in words and bool({"amount", "amt", "balance"} & set(words)))
    )
    if not labelled or _NOT_TOTAL.search(low):
        return False
    for k, w in enumerate(words):  # "Total Qty 3" counts items, but "Qty: 1  Total RM 11.60" is a total
        if _is_total_word(w) and (any(c in w for c in _COUNTS) or
                                  (k + 1 < len(words) and words[k + 1].startswith(_COUNTS))):
            return False
    return not (_GST.search(low) and not _INCL.search(low))  # "Total GST" is the tax, not the total


def _is_summary(line: str) -> bool:
    low = line.casefold()
    if re.search(r"tax\s*code|code\s*%|code\s+amount", low):
        return True
    return any(len(w) >= 5 and Levenshtein.distance(w, "summary") <= 2 for w in _words(low))


def _is_payment(line: str, amounts: list[Amount]) -> bool:
    return bool(amounts) and bool(_PAYMENT.search(line.casefold()))


_FIVE_SEN = Decimal("0.05")


def _is_round(value: Decimal) -> bool:
    return (value * 100) % 5 == 0


def _negative(line: str, amount: Amount) -> bool:
    before = line[: amount.start].rstrip(" :RMrm")
    return before.endswith(("-", "~"))


def _apply_rounding(
    lines: Sequence[str], chosen: int, end: int, total: Decimal
) -> tuple[Decimal, int] | None:
    """The rounded amount to pay, if a rounding line follows a total that is not a multiple of 5 sen."""
    if _is_round(total):
        return None
    for j in range(chosen + 1, end):
        if "round" not in lines[j].casefold():
            continue
        amounts = find_amounts(lines[j])
        if not amounts:
            continue
        a = amounts[-1].value
        if a != total and abs(a - total) <= _FIVE_SEN and _is_round(a):
            return a, j  # "ROUNDING RM 27.30" after a total of 27.28
        if 0 < a <= _FIVE_SEN:  # an adjustment such as "-0.02"
            options = (total - a, total + a) if _negative(lines[j], amounts[-1]) else (total + a, total - a)
            for value in options:
                if _is_round(value):
                    return value, j
    return None


def cash_minus_change(lines: Sequence[str]) -> Decimal | None:
    cash = change = None
    for line in lines:
        amounts = find_amounts(line)
        if not amounts:
            continue
        low = line.casefold()
        if _CHANGE.search(low):
            change = amounts[-1].value
        elif _CASH.search(low):
            cash = amounts[-1].value
    if cash is None or change is None or change > cash:
        return None
    return cash - change


def extract_total(lines: Sequence[str]) -> Extraction | None:
    candidates: list[tuple[int, Amount | None]] = []
    end = len(lines)
    for i, line in enumerate(lines):
        amounts = find_amounts(line)
        if _is_summary(line):
            end = i
            break
        if any(a for _, a in candidates) and _is_payment(line, amounts):
            end = i
            break
        if is_total_line(line):
            candidates.append((i, amounts[-1] if amounts else None))

    paid = cash_minus_change(lines)

    def result(value: Decimal, line: int, rule: str) -> Extraction:
        return Extraction(value, line, lines[line], rule, None if paid is None else paid == value)

    with_amount = [(i, a) for i, a in candidates if a is not None]
    if with_amount:
        i, amount = with_amount[-1]
        rounded = _apply_rounding(lines, i, end, amount.value)
        chosen = result(rounded[0], rounded[1], "total plus rounding") if rounded else \
            result(amount.value, i, "last total line")
        if chosen.checked is False:
            # Amounts in the totals block: on any total line, or unlabelled below the last one.
            block = {a.value: j for j, a in reversed(with_amount)}
            for j in range(i + 1, end):
                for a in find_amounts(lines[j]):
                    block.setdefault(a.value, j)
            if paid in block:
                return result(paid, block[paid], "total confirmed by cash minus change")
        return chosen
    for i, _ in candidates:
        if i + 1 < len(lines):
            below = lines[i + 1]
            amounts = find_amounts(below)
            if amounts and not _is_payment(below, amounts) and not is_total_line(below):
                return result(amounts[-1].value, i + 1, "value on the next row")
    if paid is not None:
        rows = [i for i, line in enumerate(lines) if _CHANGE.search(line.casefold()) and find_amounts(line)]
        return result(paid, rows[-1], "cash minus change")
    cards = [(i, find_amounts(line)[-1].value) for i, line in enumerate(lines)
             if _CARD.search(line.casefold()) and find_amounts(line)]
    if cards:
        return result(cards[-1][1], cards[-1][0], "card payment")
    return None


def _has_date_label(text: str) -> bool:
    return any(w in {"dt", "tarikh", "tkh"} or (len(w) == 4 and Levenshtein.distance(w, "date") <= 1)
               for w in _words(text))


_NOT_DATE = re.compile(r"exp|valid|\bdue\b|warrant|birth")


def extract_date(lines: Sequence[str]) -> Extraction | None:
    best: tuple[int, int, int, dt.date] | None = None
    for i, line in enumerate(lines):
        if _NOT_DATE.search(line.casefold()):
            continue
        for d in find_dates(line):
            key = (0 if _has_date_label(line[: d.start]) else 1, i, d.start, d.value)
            if best is None or key[:3] < best[:3]:
                best = key
    if best is None:
        return None
    labelled, i, _, value = best
    return Extraction(value, i, lines[i], "date after a label" if labelled == 0 else "first date")


def baseline_total(lines: Sequence[str]) -> Extraction | None:
    """The naive answer: the largest amount on the receipt."""
    found = [(a.value, i) for i, line in enumerate(lines) for a in find_amounts(line)]
    if not found:
        return None
    value, i = max(found, key=lambda f: (f[0], -f[1]))
    return Extraction(value, i, lines[i], "largest amount")


def baseline_date(lines: Sequence[str]) -> Extraction | None:
    """The naive answer: the first date on the receipt."""
    for i, line in enumerate(lines):
        dates = find_dates(line)
        if dates:
            return Extraction(dates[0].value, i, line, "first date")
    return None


EXTRACTORS = {
    "rules": {"total": extract_total, "date": extract_date},
    "baseline": {"total": baseline_total, "date": baseline_date},
}
