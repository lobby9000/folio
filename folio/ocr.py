"""Thin wrapper around Tesseract that keeps every word's box and confidence."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field

import numpy as np
import pytesseract

from .layout import Box, boxes_to_text


@dataclass(frozen=True)
class OcrConfig:
    """Tesseract settings.

    `psm` is the page segmentation mode: 3 lets Tesseract find the layout, 4 assumes
    a single column of lines of different sizes (a good fit for receipts) and 6 a
    single uniform block of text. `oem` 1 selects the LSTM recogniser only.
    """

    lang: str = "eng"
    psm: int = 4
    oem: int = 1

    def args(self) -> str:
        return f"--oem {self.oem} --psm {self.psm}"


@dataclass
class OcrResult:
    words: list[Box] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The words regrouped into rows, with the same rule used for the ground truth."""
        return boxes_to_text(self.words)

    @property
    def mean_confidence(self) -> float | None:
        confs = [w.conf for w in self.words if w.conf is not None]
        return sum(confs) / len(confs) if confs else None


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def run_ocr(img: np.ndarray, config: OcrConfig | None = None) -> OcrResult:
    """Run Tesseract and return one `Box` per recognised word.

    `image_to_data` reports pages, blocks, lines and words in one table. Only rows
    with text are words, and their confidence goes from 0 to 100.
    """
    config = config or OcrConfig()
    data = pytesseract.image_to_data(
        img, lang=config.lang, config=config.args(), output_type=pytesseract.Output.DICT
    )
    words = []
    for text, conf, left, top, width, height in zip(
        data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"],
        strict=True,
    ):
        text = str(text).strip()
        if not text:
            continue
        conf = float(conf)
        words.append(Box(text, left, top, left + width, top + height, conf if conf >= 0 else None))
    return OcrResult(words)
