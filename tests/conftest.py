"""Shared fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from .drawing import PAGE_LINES, render_lines


@pytest.fixture
def page() -> np.ndarray:
    return render_lines(PAGE_LINES)


@pytest.fixture
def fake_sroie(tmp_path: Path) -> Path:
    """A two-receipt folder in the SROIE layout, with invented contents."""
    root = tmp_path / "sroie"
    for sub in ("img", "box", "key"):
        (root / sub).mkdir(parents=True)
    receipts = {
        "r1": (
            ["FOLIO STORE SDN BHD", "TOTAL 12.25"],
            {"company": "FOLIO STORE SDN BHD", "date": "19/10/2026",
             "address": "1, JALAN UJI", "total": "12.25"},
        ),
        "r2": (
            ["CAFE UJI", "TOTAL 9.00"],
            {"company": "CAFE UJI", "date": "01/01/2026", "address": "2, JALAN UJI", "total": "9.00"},
        ),
    }
    for rid, (lines, key) in receipts.items():
        cv2.imwrite(str(root / "img" / f"{rid}.jpg"), render_lines(lines))
        rows = []
        for i, text in enumerate(lines):
            top, bottom = 50 + 70 * i, 50 + 70 * i + 40
            rows.append(f"50,{top},900,{top},900,{bottom},50,{bottom},{text}")
        (root / "box" / f"{rid}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (root / "key" / f"{rid}.json").write_text(json.dumps(key), encoding="utf-8")
    return root
