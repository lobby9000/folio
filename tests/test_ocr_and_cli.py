"""End-to-end checks that need the Tesseract binary. They are skipped without it."""

import pytest

from folio.cli import main
from folio.metrics import cer
from folio.ocr import OcrConfig, run_ocr, tesseract_available

from .drawing import PAGE_LINES

needs_tesseract = pytest.mark.skipif(not tesseract_available(), reason="Tesseract is not installed")


@needs_tesseract
def test_ocr_reads_a_clean_page(page):
    result = run_ocr(page, OcrConfig(psm=4))
    assert cer("\n".join(PAGE_LINES), result.text) < 0.1
    assert result.mean_confidence is not None and result.mean_confidence > 50
    assert all(w.right > w.left and w.bottom > w.top for w in result.words)


@needs_tesseract
def test_eval_command_writes_both_reports(fake_sroie, tmp_path):
    out = tmp_path / "results"
    code = main(["eval-ocr", "--data", str(fake_sroie), "--prep", "gray", "otsu", "--psm", "3",
                 "--workers", "1", "--out", str(out)])
    assert code == 0
    assert (out / "ocr_eval.csv").read_text().count("\n") == 1 + 2 * 2  # header + receipts x pipelines
    assert "| gray | 3 |" in (out / "ocr_eval.md").read_text()


@needs_tesseract
def test_extract_command_reads_total_and_date(tmp_path, capsys):
    import json

    import cv2

    from .drawing import render_lines

    image = tmp_path / "receipt.png"
    cv2.imwrite(str(image), render_lines(["FOLIO STORE", "DATE 19/10/2026", "TOTAL 12.25", "CASH 20.00",
                                          "CHANGE 7.75"]))
    assert main(["extract", str(image), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total"]["value"] == "12.25" and out["total"]["checked"] is True
    assert out["date"] == {"value": "2026-10-19", "rule": "date after a label", "row": 2,
                           "evidence": "DATE 19/10/2026", "checked": None}


@needs_tesseract
def test_eval_extract_command_writes_both_reports(fake_sroie, tmp_path):
    out = tmp_path / "results"
    code = main(["eval-extract", "--data", str(fake_sroie), "--splits", "dev", "--workers", "1",
                 "--cache", str(tmp_path / "cache"), "--out", str(out)])
    assert code == 0
    rows = (out / "extract_eval.csv").read_text().count("\n")
    assert rows == 1 + 2 * 2 * 2 * 2  # header + receipts x texts x methods x fields
    report = (out / "extract_eval.md").read_text()
    assert "| dev | OCR | rules | 2 of 2" in report and "| dev | reference | rules | 2 of 2" in report
