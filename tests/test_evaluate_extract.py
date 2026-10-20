"""Splits, the OCR cache and the extraction scoring, without running Tesseract."""

import datetime as dt
from decimal import Decimal
from pathlib import Path

from folio import cache
from folio.evaluate_extract import (
    compare_methods,
    parse_label_date,
    read_csv,
    render_markdown,
    score_receipt,
    summarize,
    summarize_checks,
    write_csv,
)
from folio.layout import Box
from folio.ocr import OcrConfig, OcrResult
from folio.sroie import Receipt, load_receipts, load_split, split_ids


def _receipt(rid, total="12.25", date="19/10/2026"):
    return Receipt(id=rid, image_path=Path(f"{rid}.jpg"), lines=(),
                   fields={"company": "", "date": date, "address": "", "total": total})


def _empty_sroie(root: Path, n: int) -> Path:
    for sub in ("img", "box", "key"):
        (root / sub).mkdir(parents=True)
    for i in range(n):
        (root / "img" / f"{i:03d}.jpg").write_bytes(b"")
        (root / "box" / f"{i:03d}.csv").write_text("")
        (root / "key" / f"{i:03d}.json").write_text("{}")
    return root


def test_splits_are_disjoint_and_dev_is_the_stage_1_sample(tmp_path):
    root = _empty_sroie(tmp_path / "sroie", 20)
    s = split_ids(root, dev_size=6, test_size=8)
    assert len(s["dev"]) == 6 and len(s["test"]) == 8
    assert not set(s["dev"]) & set(s["test"])
    assert s["dev"] == [r.id for r in load_receipts(root, limit=6, seed=0)]
    assert [r.id for r in load_split(root, "test", dev_size=6, test_size=8)] == s["test"]


def test_ocr_runs_once_per_receipt_and_then_comes_from_the_cache(fake_sroie, tmp_path, monkeypatch):
    calls = []

    def fake_ocr(image_path, prep, config):
        calls.append(image_path.name)
        return OcrResult([Box("TOTAL", 10, 10, 60, 30, 91.5), Box("12.25", 70, 10, 120, 30, None)])

    monkeypatch.setattr(cache, "ocr_image", fake_ocr)
    receipts = load_receipts(fake_sroie)
    first = cache.ocr_receipts(receipts, "gray", OcrConfig(psm=6), tmp_path / "cache", workers=1)
    second = cache.ocr_receipts(receipts, "gray", OcrConfig(psm=6), tmp_path / "cache", workers=1)
    assert sorted(calls) == ["r1.jpg", "r2.jpg"]
    assert first["r1"].text == second["r1"].text == "TOTAL 12.25"
    assert second["r1"].words[0].conf == 91.5 and second["r1"].words[1].conf is None


def test_values_are_compared_after_normalising():
    lines = ["FOLIO STORE", "Date: 19-10-26 10:22", "TOTAL RM 12.25", "CASH 20.00", "CHANGE 7.75"]
    scored = score_receipt("dev", _receipt("r1", total="RM12.25"), "ocr", lines)
    rows = {(r.method, r.field): r for r in scored}
    assert rows["rules", "total"].correct and rows["rules", "total"].checked is True
    assert rows["rules", "date"].correct and rows["rules", "date"].predicted == "2026-10-19"
    assert rows["baseline", "total"].correct is False and rows["baseline", "total"].predicted == "20.00"
    assert rows["rules", "total"].in_text and not rows["rules", "date"].verbatim  # 19-10-26 is not 19/10/2026


def test_labels_without_separators_are_read():
    assert parse_label_date("20180304") == dt.date(2018, 3, 4)
    assert parse_label_date("25032018") == dt.date(2018, 3, 25)
    assert parse_label_date("19/10/2026") == dt.date(2026, 10, 19)
    assert parse_label_date("not a date") is None


def test_empty_labels_are_left_out():
    rows = score_receipt("dev", _receipt("r1", total=""), "ocr", ["TOTAL 12.25"])
    assert all(r.correct is None for r in rows if r.field == "total")
    assert all(s.field == "date" for s in summarize(rows))


def test_summaries_pairs_and_report(tmp_path):
    rows = []
    for rid, lines in {"a": ["TOTAL 12.25", "CASH 20.00", "CHANGE 7.75"],
                       "b": ["TOTAL 12.25", "CASH 50.00"],
                       "c": ["TOTAL 9.99"]}.items():
        rows += score_receipt("test", _receipt(rid), "ocr", lines)
        rows += score_receipt("test", _receipt(rid), "reference", ["TOTAL 12.25", "19/10/2026"])
    got = {(s.source, s.method, s.field): s for s in summarize(rows)}
    assert (got["ocr", "rules", "total"].correct, got["ocr", "rules", "total"].n) == (2, 3)
    assert got["ocr", "baseline", "total"].correct == 0  # cash is the largest amount, or the total is misread
    assert got["reference", "rules", "date"].correct == 3
    pair = next(p for p in compare_methods(rows) if p.source == "ocr" and p.field == "total")
    assert (pair.only_rules, pair.only_baseline) == (2, 0)
    checks = {c.status: (c.n, c.correct) for c in summarize_checks(rows)}
    assert checks == {"agrees": (1, 1), "disagrees": (0, 0), "nothing to check": (2, 1)}
    report = render_markdown(rows, "header text")
    assert "| test | OCR | rules | 2 of 3" in report and "header text" in report
    assert "| test | c | total | `12.25` | 9.99 | yes | no | last total line | `TOTAL 9.99` |" in report
    assert "| test | last total line | 3 | 2 of 3 (67%)" in report
    write_csv(rows, tmp_path / "out.csv")
    assert read_csv(tmp_path / "out.csv") == rows
    assert Decimal(read_csv(tmp_path / "out.csv")[0].predicted) == Decimal("12.25")
