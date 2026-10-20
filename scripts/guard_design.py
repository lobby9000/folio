"""Design data for the cascade's guard against misread digits (run from the repository root).

After stage 3 the test misses showed that the cascade trusts totals whose digits OCR
misread. This script asks, on the dev and train receipts only, which cheap signal
would have caught those errors, with out-of-fold ranker scores (5 folds, the protocol
the routing threshold was chosen with). It writes results/guard_design.md.

    python scripts/guard_design.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from folio.cache import DEFAULT_CACHE, ocr_receipts  # noqa: E402
from folio.evaluate_learned import ROUTE_THRESHOLD, label  # noqa: E402
from folio.extract import _CASH, _CHANGE, extract_total  # noqa: E402
from folio.learn import make_model, pick, stack  # noqa: E402
from folio.llm import SECOND_READING, amounts_read  # noqa: E402
from folio.normalize import find_amounts, parse_amount  # noqa: E402
from folio.ocr import OcrConfig  # noqa: E402
from folio.sroie import load_split  # noqa: E402

ROOT = Path("data/sroie/data")


def digits(value) -> str:
    return re.sub(r"\D", "", f"{value}")


def word_confidence(row, value) -> float:
    """Lowest Tesseract confidence among the words on a row that carry digits of `value`."""
    want = digits(value)
    confs = [b.conf for b in row if b.conf is not None and digits(b.text) and digits(b.text) in want]
    return min(confs) if confs else 100.0


def answer_confidence(rows, line, value, rule) -> float:
    confs = [word_confidence(rows[line], value)] if line < len(rows) else [100.0]
    if rule == "cash minus change":  # that answer rests on the cash and change rows
        for row in rows:
            text = " ".join(b.text for b in row).casefold()
            if (_CASH.search(text) or _CHANGE.search(text)) and find_amounts(text):
                confs.append(word_confidence(row, find_amounts(text)[-1].value))
    return min(confs)


def main() -> int:
    pool = [r for s in ("dev", "train") for r in load_split(ROOT, s) if r.fields["total"]]
    gray = ocr_receipts(pool, cache_dir=DEFAULT_CACHE, workers=2)
    second = ocr_receipts(pool, prep=SECOND_READING, config=OcrConfig(psm=6), cache_dir=DEFAULT_CACHE,
                          workers=2)
    labelled = [label(r, gray[r.id]) for r in pool]

    order = np.random.default_rng(0).permutation(len(labelled))
    oof = {}
    for f in range(5):
        X, y, _ = stack([labelled[k] for j, k in enumerate(order) if j % 5 != f])
        model = make_model("logistic").fit(X, y)
        for k in (k for j, k in enumerate(order) if j % 5 == f):
            oof[k] = pick(model, labelled[k])

    trusted = []
    for k, receipt in enumerate(pool):
        gold = parse_amount(receipt.fields["total"])
        lines, rows = gray[receipt.id].text.splitlines(), gray[receipt.id].rows
        rules = extract_total(lines)
        got = oof.get(k)
        if rules is not None and rules.checked:
            route, value, line, rule = "cash", rules.value, rules.line, rules.rule
        elif got is not None and got[1] >= ROUTE_THRESHOLD:
            route, value, line, rule = "ranker", got[0].value, got[0].row, "ranker"
        else:
            continue
        ok = value == gold
        kind = "" if ok else ("misread digit" if Levenshtein.distance(digits(value), digits(gold)) <= 2
                              else "wrong amount")
        trusted.append(dict(id=receipt.id, route=route, rule=rule, ok=ok, kind=kind, value=value, gold=gold,
                            conf=answer_confidence(rows, line, value, rule),
                            seen_twice=value in amounts_read(second[receipt.id].text)))

    wrong = [t for t in trusted if not t["ok"]]
    out = ["# Design data for the guard against misread digits", "",
           f"{len(pool)} dev and train receipts, out-of-fold ranker scores. The cascade trusts the total on "
           f"{len(trusted)} of them (cash check or ranker score at least {ROUTE_THRESHOLD}) and is wrong on "
           f"{len(wrong)}: {dict(Counter(t['kind'] for t in wrong))}. The test split is not read here.", "",
           "| Receipt | Route | Rule | Answer | Label | Word confidence | Second pass read it |",
           "|---|---|---|---|---|---|---|"]
    for t in wrong:
        seen = "yes" if t["seen_twice"] else "no"
        out.append(f"| {t['id']} | {t['route']} | {t['rule']} | {t['value']} | {t['gold']} | {t['conf']:.0f} "
                   f"| {seen} |")
    out += ["", "| Signal that sends a trusted total back to the LLM | Errors caught "
            "| Right answers sent back |", "|---|---|---|"]

    def row(name, flag):
        caught = sum(flag(t) for t in wrong)
        extra = sum(flag(t) for t in trusted if t["ok"])
        out.append(f"| {name} | {caught} of {len(wrong)} | {extra} |")

    def self_confirmed(t):
        return t["route"] == "cash" and t["rule"] == "cash minus change"

    for limit in (50, 70, 80, 90):
        row(f"OCR word confidence below {limit}", lambda t, limit=limit: t["conf"] < limit)
    row(f"the second OCR pass (`{SECOND_READING}`) did not read it", lambda t: not t["seen_twice"])
    row("the cash check confirmed its own fallback", self_confirmed)
    row("either of the last two (the guard)", lambda t: not t["seen_twice"] or self_confirmed(t))
    Path("results/guard_design.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
