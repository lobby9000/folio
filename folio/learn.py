"""A learned extractor for the total: score every amount on the receipt, keep the best.

Stage 2's rules weigh the evidence by hand. Here every amount on the receipt becomes a
candidate described by features: the words on its row and on the rows around it,
where the row sits on the page and how tall it is, how often the same value repeats,
whether it equals cash tendered minus change, and whether the rules picked it. A
model fitted on labelled receipts learns how much each of those counts, and the
candidate with the highest score is the answer. Its score doubles as a confidence,
which decides when a receipt is worth sending to an LLM or to a person.

Two models are compared: logistic regression, whose weights can be read and saved as
plain JSON, and gradient-boosted trees, which can learn interactions between
features. Both see exactly the same features.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np

from .extract import (
    _CARD,
    _CASH,
    _CHANGE,
    Extraction,
    _is_payment,
    _is_summary,
    _is_total_word,
    _words,
    cash_minus_change,
    extract_total,
    is_total_line,
)
from .layout import Box
from .normalize import find_amounts

_KEYWORDS = {
    "kw_total_word": lambda low, words: any(_is_total_word(w) for w in words),
    "kw_strong": lambda low, words: bool({"payable", "nett", "grand", "rounded", "final", "net"} & set(words))
    or ("after" in words and any(w.startswith(("round", "adj")) for w in words)),
    "kw_sub": lambda low, words: bool(re.search(r"sub\W?t", low)),
    "kw_count": lambda low, words: any(w.startswith(("qty", "quantit", "item", "count", "pcs"))
                                       for w in words),
    "kw_excl": lambda low, words: bool(re.search(r"\bexc|before", low)),
    "kw_gst": lambda low, words: bool(re.search(r"gst|gs7|&st", low)),
    "kw_incl": lambda low, words: bool(re.search(r"\binc(?:l|\b|\.|\])", low)),
    "kw_tax": lambda low, words: bool(re.search(r"\btax\b", low)),
    "kw_cash": lambda low, words: bool(_CASH.search(low)),
    "kw_change": lambda low, words: bool(_CHANGE.search(low)),
    "kw_card": lambda low, words: bool(_CARD.search(low)),
    "kw_round": lambda low, words: "round" in low,
    "kw_saving": lambda low, words: bool(re.search(r"saving|discount|disc\b", low)),
    "kw_amount": lambda low, words: bool({"amount", "amt"} & set(words)),
    "kw_currency": lambda low, words: bool(re.search(r"\brm|\$|myr", low)),
}

FEATURES = [
    *_KEYWORDS,
    "row_is_total_line",
    "prev_is_total_line",
    "next_is_total_line",
    "next_is_payment",
    "after_payment",
    "after_summary",
    "row_position",
    "rows_to_end",
    "log_value",
    "value_rank",
    "is_largest",
    "repeats",
    "is_round",
    "equals_paid",
    "has_paid",
    "last_on_row",
    "amounts_on_row",
    "one_decimal",
    "rules_value",
    "rules_row",
    "value_on_total_line",
    "value_on_strong_line",
    "value_on_payment_line",
    "row_height",
    "row_right",
    "row_left",
    "row_conf",
]


@dataclass(frozen=True)
class Candidate:
    row: int
    value: Decimal
    text: str


def _layout(rows: Sequence[Sequence[Box]]) -> tuple[float, float, float]:
    boxes = [b for row in rows for b in row]
    heights = [b.height for b in boxes] or [1.0]
    left = min((b.left for b in boxes), default=0.0)
    right = max((b.right for b in boxes), default=1.0)
    return statistics.median(heights), left, max(right - left, 1.0)


def candidates(rows: Sequence[Sequence[Box]]) -> tuple[list[Candidate], np.ndarray]:
    """Every amount on the receipt, with its feature vector (columns in `FEATURES` order)."""
    lines = [" ".join(b.text for b in row) for row in rows]
    lows = [line.casefold() for line in lines]
    amounts = [find_amounts(line) for line in lines]
    total_lines = [is_total_line(line) for line in lines]
    payments = [_is_payment(line, a) for line, a in zip(lines, amounts, strict=True)]
    summaries = [_is_summary(line) for line in lines]
    first_payment = next((i for i, p in enumerate(payments) if p and any(total_lines[:i])), len(lines))
    first_summary = next((i for i, s in enumerate(summaries) if s), len(lines))
    rules = extract_total(lines)
    paid = cash_minus_change(lines)
    median_h, page_left, page_width = _layout(rows)
    all_values = [a.value for found in amounts for a in found]
    distinct = sorted(set(all_values), reverse=True)
    n = len(lines)

    # Evidence pooled per value: the same amount printed on several rows (a total line,
    # the cash line, a rounding line) is one answer seen several times.
    strong = [_KEYWORDS["kw_strong"](low, _words(low)) for low in lows]
    on_total, on_strong, on_payment = set(), set(), set()
    for i, found in enumerate(amounts):
        for a in found:
            if total_lines[i]:
                on_total.add(a.value)
            if total_lines[i] and strong[i]:
                on_strong.add(a.value)
            if payments[i]:
                on_payment.add(a.value)

    found_candidates: list[Candidate] = []
    feats: list[list[float]] = []
    for i, (line, low, found) in enumerate(zip(lines, lows, amounts, strict=True)):
        if not found:
            continue
        words = _words(low)
        row = rows[i]
        confs = [b.conf for b in row if b.conf is not None]
        row_f = [float(fn(low, words)) for fn in _KEYWORDS.values()] + [
            float(total_lines[i]),
            float(i > 0 and total_lines[i - 1]),
            float(i + 1 < n and total_lines[i + 1]),
            float(i + 1 < n and payments[i + 1]),
            float(i > first_payment),
            float(i > first_summary),
            i / max(n - 1, 1),
            (n - 1 - i) / max(n, 1),
        ]
        layout_f = [
            statistics.median(b.height for b in row) / median_h,
            (max(b.right for b in row) - page_left) / page_width,
            (min(b.left for b in row) - page_left) / page_width,
            (sum(confs) / len(confs) / 100) if confs else 1.0,
        ]
        for k, a in enumerate(found):
            digits = a.raw.replace(" ", "")
            value_f = [
                math.log1p(float(a.value)),
                distinct.index(a.value) / max(len(distinct) - 1, 1),
                float(a.value == distinct[0]),
                float(all_values.count(a.value) - 1),
                float((a.value * 100) % 5 == 0),
                float(paid is not None and a.value == paid),
                float(paid is not None),
                float(k == len(found) - 1),
                float(len(found)),
                float(bool(re.search(r"[.,]\d$", digits))),
                float(rules is not None and a.value == rules.value),
                float(rules is not None and i == rules.line),
                float(a.value in on_total),
                float(a.value in on_strong),
                float(a.value in on_payment),
            ]
            found_candidates.append(Candidate(i, a.value, line))
            feats.append(row_f + value_f + layout_f)
    matrix = np.array(feats, dtype=float) if feats else np.zeros((0, len(FEATURES)))
    return found_candidates, matrix


@dataclass
class LabelledReceipt:
    receipt: str
    candidates: list[Candidate]
    features: np.ndarray
    gold: Decimal | None

    @property
    def labels(self) -> np.ndarray:
        return np.array([c.value == self.gold for c in self.candidates], dtype=int)

    @property
    def reachable(self) -> bool:
        """The right value is among the candidates, so a ranker could find it."""
        return self.gold is not None and any(c.value == self.gold for c in self.candidates)


def make_model(kind: str):
    """A fresh, unfitted model: "logistic" or "boosting"."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    if kind == "boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                              l2_regularization=1.0, random_state=0)
    raise ValueError(f"unknown model kind {kind!r}")


def stack(receipts: Sequence[LabelledReceipt]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features, labels and the receipt index of each candidate, for fitting."""
    used = [r for r in receipts if len(r.candidates)]
    X = np.vstack([r.features for r in used])
    y = np.concatenate([r.labels for r in used])
    groups = np.concatenate([[k] * len(r.candidates) for k, r in enumerate(used)])
    return X, y, groups


def pick(model, receipt: LabelledReceipt) -> tuple[Candidate, float] | None:
    """The highest-scoring candidate and its score (a probability between 0 and 1)."""
    if not receipt.candidates:
        return None
    scores = model.predict_proba(receipt.features)[:, 1]
    best = int(np.argmax(scores))
    return receipt.candidates[best], float(scores[best])


def receipt_accuracy(model, receipts: Sequence[LabelledReceipt]) -> float:
    hits = 0
    for r in receipts:
        got = pick(model, r)
        hits += got is not None and got[0].value == r.gold
    return hits / len(receipts) if receipts else float("nan")


def cross_validate(
    kind: str, receipts: Sequence[LabelledReceipt], folds: int = 5, seed: int = 0
) -> list[float]:
    """Receipt-level accuracy of each fold, holding out whole receipts so no receipt is
    seen in training and scoring at once."""
    order = np.random.default_rng(seed).permutation(len(receipts))
    scores = []
    for f in range(folds):
        held = [receipts[k] for j, k in enumerate(order) if j % folds == f]
        fit_on = [receipts[k] for j, k in enumerate(order) if j % folds != f]
        X, y, _ = stack(fit_on)
        model = make_model(kind).fit(X, y)
        scores.append(receipt_accuracy(model, held))
    return scores


class TotalRanker:
    """A fitted model plus the steps to turn a receipt's rows into an answer."""

    def __init__(self, model, kind: str):
        self.model = model
        self.kind = kind

    @classmethod
    def fit(cls, receipts: Sequence[LabelledReceipt], kind: str = "logistic") -> TotalRanker:
        X, y, _ = stack(receipts)
        return cls(make_model(kind).fit(X, y), kind)

    def extract(self, rows: Sequence[Sequence[Box]]) -> Extraction | None:
        found, X = candidates(rows)
        if not found:
            return None
        got = pick(self.model, LabelledReceipt("", found, X, None))
        assert got is not None
        cand, score = got
        lines = [" ".join(b.text for b in row) for row in rows]
        paid = cash_minus_change(lines)
        return Extraction(cand.value, cand.row, cand.text, f"{self.kind} ranker",
                          None if paid is None else paid == cand.value, score)

    def weights(self) -> dict[str, float]:
        """Standardised weights of the logistic model, largest first: how much one
        standard deviation of each feature moves the log-odds of being the total."""
        if self.kind != "logistic":
            raise ValueError("only the logistic model has readable weights")
        coef = self.model[-1].coef_[0]
        return dict(sorted(zip(FEATURES, map(float, coef), strict=True), key=lambda kv: -abs(kv[1])))

    def save(self, path: Path) -> None:
        """Logistic models only: scaler and weights as JSON, readable and safe to load."""
        if self.kind != "logistic":
            raise ValueError("only the logistic model is saved as JSON")
        scaler, lr = self.model[0], self.model[-1]
        data = {"features": FEATURES, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TotalRanker:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["features"] != FEATURES:
            raise ValueError("the saved model was trained with different features")
        return cls(_JsonLogistic(data), "logistic")


class _JsonLogistic:
    """The saved logistic model, scored with NumPy alone."""

    def __init__(self, data: dict):
        self.mean = np.array(data["mean"])
        self.scale = np.array(data["scale"])
        self.coef = np.array(data["coef"])
        self.intercept = data["intercept"]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = ((X - self.mean) / self.scale) @ self.coef + self.intercept
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])
