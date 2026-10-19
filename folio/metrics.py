"""Text-recognition and field metrics.

* CER and WER: edit distance between the reference and the OCR output, divided by
  the length of the reference, in characters or in words. Edit distance counts the
  substitutions, deletions and insertions needed to turn one text into the other.
* Bag-of-words precision, recall and F1: how many reference words the OCR produced,
  ignoring order. This is close to how the SROIE OCR task was scored.
* Field found: whether a key value, such as the receipt total, appears verbatim in
  the OCR output. One wrong digit in the total matters more than ten in a footer,
  so OCR quality is also measured where it counts.

Texts are compared case-insensitively with whitespace collapsed, because the SROIE
transcriptions are mostly uppercase even where the print is not.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein


def normalize(text: str) -> str:
    """Collapse whitespace and fold case."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate. It can exceed 1 when the OCR invents a lot of text."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: the same edit distance over lists of words."""
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float


def bag_of_words(reference: str, hypothesis: str) -> PRF:
    """Precision, recall and F1 over word multisets, ignoring order."""
    ref = Counter(normalize(reference).split())
    hyp = Counter(normalize(hypothesis).split())
    matched = sum((ref & hyp).values())
    precision = matched / sum(hyp.values()) if hyp else 0.0
    recall = matched / sum(ref.values()) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision, recall, f1)


def field_found(value: str, text: str) -> bool:
    """True if `value` appears in `text` as a whole token sequence.

    Digits and letters may not touch the match on either side, so a total of 9.00
    is not found inside 19.00, and 25/12/2018 is not found inside 125/12/20180.
    """
    value, text = normalize(value), normalize(text)
    if not value:
        return False
    pattern = r"(?<![0-9a-z])" + re.escape(value) + r"(?![0-9a-z])"
    return re.search(pattern, text) is not None


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Better than the normal one near 0 or 1."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
