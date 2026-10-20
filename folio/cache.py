"""OCR once, reuse many times.

Extraction rules change far more often than the OCR settings, so the OCR output of
each receipt is stored as JSON (every word with its box and confidence) under
`cache/ocr/<preprocessing>-psm<n>-<lang>/<receipt id>.json`. Rerunning an
extraction experiment then takes seconds instead of minutes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from .ocr import OcrConfig, OcrResult, run_ocr
from .preprocess import PIPELINES
from .sroie import Receipt

DEFAULT_CACHE = Path("cache/ocr")


def cache_path(cache_dir: Path, receipt_id: str, prep: str, config: OcrConfig) -> Path:
    return Path(cache_dir) / f"{prep}-psm{config.psm}-{config.lang}" / f"{receipt_id}.json"


def ocr_image(image_path: Path, prep: str, config: OcrConfig) -> OcrResult:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    return run_ocr(PIPELINES[prep](img), config)


def _ocr_job(job: tuple[str, Path, str, OcrConfig, Path]) -> str:
    receipt_id, image_path, prep, config, path = job
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    result = ocr_image(image_path, prep, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    tmp.replace(path)  # atomic: an interrupted run never leaves half a file
    return receipt_id


def ocr_receipts(
    receipts: Sequence[Receipt],
    prep: str = "gray",
    config: OcrConfig | None = None,
    cache_dir: Path = DEFAULT_CACHE,
    workers: int = 1,
) -> dict[str, OcrResult]:
    """OCR every receipt that is not cached yet, then load all of them from the cache."""
    config = config or OcrConfig(psm=6)
    missing = [
        (r.id, r.image_path, prep, config, cache_path(cache_dir, r.id, prep, config))
        for r in receipts
        if not cache_path(cache_dir, r.id, prep, config).exists()
    ]
    if missing:
        if workers <= 1:
            for job in missing:
                _ocr_job(job)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_ocr_job, missing))
    return {r.id: _load(cache_path(cache_dir, r.id, prep, config)) for r in receipts}


def _load(path: Path) -> OcrResult:
    return OcrResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
