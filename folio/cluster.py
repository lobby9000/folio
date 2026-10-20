"""Receipt layouts without labels.

Receipts from the same shop share a layout: the header, the address, the footer and
the columns repeat. Finding those groups without labels is useful twice over. A new
layout is where cheap extractors are most likely to fail, so it can be sent to a
stronger one. And groups give honest data splits: folds that keep a whole layout on
one side cannot reward a model for memorising it.

A receipt is represented in one of four ways (its OCR text, the start of it, where
its words sit on the page, or a thumbnail of the scan), compared by cosine distance,
and grouped by agglomerative clustering with average linkage, cut at a distance
threshold so the number of groups is not fixed in advance. The merchant names in the
SROIE labels are used only to score the groups, never to form them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .ocr import OcrResult
from .sroie import Receipt

# Branches of one chain, and typos in the labels, that are the same merchant for scoring.
CHAINS = ("MR D I Y", "AEON CO", "GARDENIA BAKERIES", "DOMINO S PIZZA", "GL HANDICRAFT", "WESTERN EASTERN",
          "BHPETROL", "NADEJE")

KINDS = ("OCR text, words", "first rows, characters", "word positions", "thumbnail")
HEADER_ROWS = 6


def merchant(company: str) -> str:
    """A merchant key from a SROIE company label: case, punctuation and legal suffixes removed,
    and the branches of a few chains merged (`MR D.I.Y. (KUCHAI) SDN BHD` is `MR D I Y`)."""
    key = re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", company.upper())).strip()
    key = re.sub(r"\s+", " ", re.sub(r"\b(SDN BHD|S B|BHD|SB)\b", "", key)).strip()
    return next((chain for chain in CHAINS if key.startswith(chain)), key)


def _position_grid(ocr: OcrResult, rows: int = 24, cols: int = 8) -> np.ndarray:
    """How many words fall in each cell of a grid laid over the text area."""
    words = [w for w in ocr.words if w.text.strip()]
    grid = np.zeros((rows, cols))
    if not words:
        return grid.ravel()
    left, top = min(w.left for w in words), min(w.top for w in words)
    width = max(max(w.right for w in words) - left, 1.0)
    height = max(max(w.bottom for w in words) - top, 1.0)
    for w in words:
        y = int((w.y_center - top) / height * rows)
        x = int(((w.left + w.right) / 2 - left) / width * cols)
        grid[min(max(y, 0), rows - 1), min(max(x, 0), cols - 1)] += 1
    return grid.ravel()


def _thumbnail(path, width: int = 32, height: int = 96) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if Path(path).exists() else None
    if img is None:
        return np.zeros(width * height)
    return 255.0 - cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA).astype(float).ravel()


def vectorizer(kind: str):
    """The fitted part of a text representation, or None for the others."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if kind == "OCR text, words":
        return TfidfVectorizer(sublinear_tf=True, min_df=1)
    if kind == "first rows, characters":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    return None


def _texts(kind: str, ocrs: Sequence[OcrResult]) -> list[str]:
    if kind == "first rows, characters":
        return ["\n".join(o.text.splitlines()[:HEADER_ROWS]) for o in ocrs]
    return [o.text for o in ocrs]


def represent(kind: str, receipts: Sequence[Receipt], ocrs: Sequence[OcrResult], fitted=None):
    """One row per receipt. Text representations are fitted on these receipts unless an
    already fitted vectorizer is given (to place new receipts in a known space)."""
    if kind in ("OCR text, words", "first rows, characters"):
        vec = fitted if fitted is not None else vectorizer(kind).fit(_texts(kind, ocrs))
        return vec.transform(_texts(kind, ocrs)), vec
    if kind == "word positions":
        return np.array([_position_grid(o) for o in ocrs]), None
    if kind == "thumbnail":
        return np.array([_thumbnail(r.image_path) for r in receipts]), None
    raise ValueError(f"unknown representation {kind!r}")


def distances(X, Y=None) -> np.ndarray:
    """Cosine distances; an empty receipt is at distance 1 from everything."""
    from sklearn.metrics.pairwise import cosine_similarity

    return np.clip(1.0 - cosine_similarity(X, Y), 0.0, 2.0)


def cluster(D: np.ndarray, threshold: float) -> np.ndarray:
    """Groups from average-linkage agglomerative clustering, cut at `threshold`."""
    from sklearn.cluster import AgglomerativeClustering

    if len(D) < 2:
        return np.zeros(len(D), dtype=int)
    return AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="precomputed",
                                   linkage="average").fit_predict(D)


def scores(truth: Sequence[str], groups: Sequence[int]) -> dict[str, float]:
    """How well the groups match the merchants. Homogeneity: each group holds one merchant.
    Completeness: each merchant sits in one group. Pairwise precision and recall: of the
    pairs of receipts put together, how many share a merchant, and of the pairs that share
    a merchant, how many were put together."""
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        homogeneity_completeness_v_measure,
    )

    truth, groups = np.asarray(truth), np.asarray(groups)
    h, c, _ = homogeneity_completeness_v_measure(truth, groups)
    same_t = truth[:, None] == truth[None, :]
    same_g = groups[:, None] == groups[None, :]
    upper = np.triu(np.ones_like(same_t, dtype=bool), k=1)
    together, should = (same_g & upper).sum(), (same_t & upper).sum()
    both = (same_g & same_t & upper).sum()
    return {
        "ami": float(adjusted_mutual_info_score(truth, groups)),
        "ari": float(adjusted_rand_score(truth, groups)),
        "homogeneity": float(h),
        "completeness": float(c),
        "groups": int(len(set(groups.tolist()))),
        "merchants": int(len(set(truth.tolist()))),
        "pair_precision": float(both / together) if together else float("nan"),
        "pair_recall": float(both / should) if should else float("nan"),
    }


def nearest_distance(D: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """For each row, the distance to its nearest column (another receipt when `exclude_self`)."""
    D = D.copy()
    if exclude_self:
        np.fill_diagonal(D, np.inf)
    return D.min(axis=1)
