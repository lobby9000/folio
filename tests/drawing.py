"""Drawing helpers for tests: pages are rendered here, so no dataset files live in the repo."""

from __future__ import annotations

import cv2
import numpy as np

PAGE_LINES = [
    "FOLIO STORE SDN BHD",
    "INVOICE NO 4417",
    "DATE 19/10/2026",
    "COFFEE 2 X 4.50",
    "BAGEL 1 X 3.25",
    "TOTAL 12.25",
]


def render_lines(lines: list[str], width: int = 1100, line_height: int = 70, margin: int = 50) -> np.ndarray:
    """Black text on a white page, one line per entry, large enough for Tesseract."""
    height = 2 * margin + line_height * len(lines)
    img = np.full((height, width), 255, np.uint8)
    for i, text in enumerate(lines):
        y = margin + (i + 1) * line_height - 18
        cv2.putText(img, text, (margin, y), cv2.FONT_HERSHEY_DUPLEX, 1.4, 0, 2, cv2.LINE_AA)
    return img
