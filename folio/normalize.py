"""Read amounts and dates out of noisy receipt text.

Two receipts can write the same total as `RM 5.90`, `5,90`, `$5.90` or, after OCR,
`RM5. 90`. Comparing strings would call those different, so both the extractor and
the evaluation turn text into values first: a `Decimal` for money and a
`datetime.date` for dates.

The tolerances are the ones seen in the development receipts and nothing more
exotic: a currency prefix glued to the number, a comma as decimal separator, a
stray space around the separator, a dropped trailing zero (`174.9`), a missing
leading zero (`.02`) and the letter O read in place of a zero. Anything else,
such as `S` for `5` or a missing decimal point, is left as an OCR error.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from decimal import Decimal

# Currency marks glued to a number, including the usual OCR misreadings of "RM"
# (RN, RH, RY) and the separators that end up between mark and digits.
_CURRENCY = re.compile(r"(?i)(?<![a-z])(?:rm|rn|rh|ry|myr)[\s_:]*(?=[\dOo.])|\$\s*(?=[\dOo.])")

_DIGIT = "[0-9Oo]"
_AMOUNT = re.compile(
    r"(?<![\w.,])"
    rf"(?:(?P<int>\d{{1,3}}(?:,\d{{3}})+|{_DIGIT}+)\s?(?P<sep>[.,])\s?|\.)"
    rf"(?P<dec>{_DIGIT}{{1,2}})"
    r"(?![\w%]|[.,]\d)"
)


@dataclass(frozen=True)
class Amount:
    value: Decimal
    start: int
    end: int
    raw: str


def _digits(text: str) -> str:
    return text.replace("O", "0").replace("o", "0")


def find_amounts(text: str) -> list[Amount]:
    """Every money amount in a line, left to right. Integers without decimals are skipped
    on purpose: on a receipt they are far more often quantities or codes than money."""
    cleaned = _CURRENCY.sub(lambda m: " " * len(m.group(0)), text)
    found = []
    for m in _AMOUNT.finditer(cleaned):
        raw = m.group(0)
        if not any(c.isdigit() for c in raw):  # "O. OO" is not an amount
            continue
        whole = _digits((m.group("int") or "0").replace(",", ""))
        decimals = _digits(m.group("dec")).ljust(2, "0")
        found.append(Amount(Decimal(f"{int(whole)}.{decimals}"), m.start(), m.end(), text[m.start():m.end()]))
    return found


def parse_amount(text: str) -> Decimal | None:
    """The value of a labelled amount such as `RM 34.80` or `$8.90`.

    Used on the SROIE labels, which are clean, so a bare integer counts too."""
    amounts = find_amounts(text)
    if amounts:
        return amounts[-1].value
    m = re.fullmatch(r"\s*(?:RM|\$)?\s*(\d+)\s*", text, re.I)
    return Decimal(m.group(1)) if m else None


MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH = (
    r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?![a-z])"
)
_DATE_PATTERNS = [
    # 19/10/2018, 18-10-17, 11.02.18, 4/2/2017 (day first, the convention in Malaysia)
    ("dmy", re.compile(
        rf"(?<![\w/.\-])(?P<d>{_DIGIT}{{1,2}})(?P<sep>[/\-.])(?P<m>{_DIGIT}{{1,2}})(?P=sep)"
        rf"(?P<y>{_DIGIT}{{4}}|{_DIGIT}{{2}})(?![\w/]|\.\d)")),
    # 2018-04-06
    ("ymd", re.compile(
        r"(?<![\w/.\-])(?P<y>\d{4})(?P<sep>[/\-.])(?P<m>\d{1,2})(?P=sep)(?P<d>\d{1,2})(?![\w/]|[.\-]\d)")),
    # 10 Mar 2018, 05-Aug-2017, 25/FEB/2017, 11 December, 2017, 15 Jun 18
    ("d mon y", re.compile(
        rf"(?i)(?<![\w])(?P<d>\d{{1,2}})[\s\-/.]*{_MONTH}\.?[\s\-/.,]*(?P<y>\d{{4}}|\d{{2}})(?![\w])")),
    # OCT 3, 2016
    ("mon d y", re.compile(rf"(?i)(?<![\w]){_MONTH}\.?\s+(?P<d>\d{{1,2}}),?\s+(?P<y>\d{{4}})(?![\w])")),
]


@dataclass(frozen=True)
class DateMatch:
    value: dt.date
    start: int
    end: int
    raw: str


def _make_date(year: int, month: int, day: int, years: tuple[int, int]) -> dt.date | None:
    if year < 100:
        year += 2000
    if not years[0] <= year <= years[1]:
        return None
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def find_dates(text: str, years: tuple[int, int] = (2000, 2030)) -> list[DateMatch]:
    """Every plausible date in a line, left to right.

    Numeric dates are read day first. If that is impossible (a month above 12) and the
    separator is a slash, the month-first reading is tried, so `12/25/2018` still
    parses while a code such as `2.13.12` does not become a date. Years outside
    `years` are rejected, which removes OCR misreadings such as `9018-04-06`."""
    found: list[DateMatch] = []
    taken: list[range] = []
    for kind, pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() in r or m.end() - 1 in r for r in taken):
                continue
            g = m.groupdict()
            year = int(_digits(g["y"]))
            if "mon" in g and g["mon"]:
                value = _make_date(year, MONTHS[g["mon"][:3].lower()], int(g["d"]), years)
            else:
                day, month = int(_digits(g["d"])), int(_digits(g["m"]))
                value = _make_date(year, month, day, years)
                if value is None and kind == "dmy" and g["sep"] == "/":
                    value = _make_date(year, day, month, years)
            if value is not None:
                found.append(DateMatch(value, m.start(), m.end(), m.group(0)))
                taken.append(range(m.start(), m.end()))
    return sorted(found, key=lambda d: d.start)


def parse_date(text: str) -> dt.date | None:
    """The first date in `text`, used on the SROIE labels."""
    dates = find_dates(text)
    return dates[0].value if dates else None
