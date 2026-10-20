"""The learned ranker on invented receipts: features, fitting, saving and the stage 3 report."""

import random
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from folio.evaluate_learned import ScoredReceipt, render_markdown, route_summary, run
from folio.layout import Box, group_rows
from folio.learn import FEATURES, LabelledReceipt, TotalRanker, candidates, cross_validate
from folio.ocr import OcrResult
from folio.sroie import Receipt


def receipt_rows(rows: list[tuple[str, str | None]], height: float = 30) -> list[list[Box]]:
    """A receipt as rows of a label on the left and an optional amount on the right."""
    boxes = []
    for i, (label, value) in enumerate(rows):
        top = 50 + 45 * i
        boxes.append(Box(label, 20, top, 20 + 12 * len(label), top + height, 90.0))
        if value is not None:
            boxes.append(Box(value, 600, top, 600 + 12 * len(value), top + height, 85.0))
    return group_rows(boxes)


def shop(rng: random.Random) -> tuple[list[list[Box]], Decimal]:
    items = [Decimal(rng.randint(150, 2500)) / 100 for _ in range(rng.randint(2, 5))]
    subtotal = sum(items)
    total = (subtotal * 2).quantize(Decimal("0.1")) / 2  # rounded to 5 sen
    cash = Decimal(int(total) + rng.choice([1, 5, 10, 20]))
    rows = [("FOLIO MART SDN BHD", None), ("RECEIPT", None)]
    rows += [(f"ITEM {k}", f"{v:.2f}") for k, v in enumerate(items)]
    rows += [("SUB TOTAL", f"{subtotal:.2f}"), ("TOTAL", f"{total:.2f}"), ("CASH", f"{cash:.2f}"),
             ("CHANGE", f"{cash - total:.2f}"), ("THANK YOU", None)]
    return receipt_rows(rows), total


def labelled_shops(n: int, seed: int = 0) -> list[LabelledReceipt]:
    rng = random.Random(seed)
    out = []
    for k in range(n):
        rows, total = shop(rng)
        found, X = candidates(rows)
        out.append(LabelledReceipt(f"r{k}", found, X, total))
    return out


def test_every_amount_is_a_candidate_with_one_value_per_feature():
    rows = receipt_rows([("TOTAL", "12.25"), ("CASH", "20.00"), ("CHANGE", "7.75")])
    found, X = candidates(rows)
    assert [c.value for c in found] == [Decimal("12.25"), Decimal("20.00"), Decimal("7.75")]
    assert X.shape == (3, len(FEATURES))
    col = {name: X[:, k] for k, name in enumerate(FEATURES)}
    assert list(col["equals_paid"]) == [1, 0, 0]  # 20.00 - 7.75
    assert list(col["is_largest"]) == [0, 1, 0]
    assert list(col["kw_cash"]) == [0, 1, 0] and list(col["row_is_total_line"]) == [1, 0, 0]
    assert list(col["rules_value"]) == [1, 0, 0]


def test_a_receipt_without_amounts_has_no_candidates():
    found, X = candidates(receipt_rows([("THANK YOU", None)]))
    assert found == [] and X.shape == (0, len(FEATURES))
    assert TotalRanker.fit(labelled_shops(20), "logistic").extract(receipt_rows([("HELLO", None)])) is None


@pytest.mark.parametrize("kind", ["logistic", "boosting"])
def test_ranker_learns_to_pick_the_total(kind):
    ranker = TotalRanker.fit(labelled_shops(40), kind)
    rows, total = shop(random.Random(99))
    got = ranker.extract(rows)
    assert got.value == total and got.rule == f"{kind} ranker"
    assert got.checked is True and 0.5 < got.score <= 1


def test_cross_validation_scores_whole_receipts():
    scores = cross_validate("logistic", labelled_shops(30), folds=3)
    assert len(scores) == 3 and all(0.9 <= s <= 1 for s in scores)


def test_saved_model_scores_exactly_like_the_fitted_one(tmp_path):
    receipts = labelled_shops(30)
    ranker = TotalRanker.fit(receipts, "logistic")
    ranker.save(tmp_path / "ranker.json")
    loaded = TotalRanker.load(tmp_path / "ranker.json")
    X = np.vstack([r.features for r in receipts[:5]])
    assert np.allclose(ranker.model.predict_proba(X), loaded.model.predict_proba(X))
    assert set(ranker.weights()) == set(FEATURES)
    with pytest.raises(ValueError):
        TotalRanker.fit(receipts, "boosting").save(tmp_path / "boosting.json")


def _receipt_and_ocr(rid: str, rng: random.Random) -> tuple[Receipt, OcrResult]:
    rows, total = shop(rng)
    words = [b for row in rows for b in row]
    fields = {"company": "", "date": "", "address": "", "total": f"{total}"}
    receipt = Receipt(rid, Path(f"{rid}.jpg"), (), fields)
    return receipt, OcrResult(words)


def test_stage_3_protocol_end_to_end():
    rng = random.Random(5)
    pool = [(*_receipt_and_ocr(f"p{k}", rng), "train") for k in range(30)]
    test = [_receipt_and_ocr(f"t{k}", rng) for k in range(10)]
    report, ranker = run(pool, test, folds=3, repeats=1)
    assert report.pool_size == 30 and report.chosen in ("logistic", "boosting")
    assert report.counts()["n"] == 10 and report.counts()["ranker"] == 10
    md = render_markdown(report, "header")
    assert "## Held-out test, scored once" in md and "| Learned ranker" in md
    assert "## Routing for stage 3" in md


def test_route_summary_splits_receipts_three_ways():
    def row(checked, score, rules_ok, ranker_ok):
        return ScoredReceipt("r", "1.00", "", rules_ok, checked, "", ranker_ok, score, True, "")

    rows = [row(True, 0.1, True, False), row(None, 0.99, False, True), row(False, 0.5, False, True)]
    assert route_summary(rows, 0.95) == {"confirmed": 1, "confirmed_ok": 1, "confident": 1, "confident_ok": 1,
                                         "rest": 1, "rest_ok": 1}
