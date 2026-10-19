"""Loader for the SROIE 2019 receipts (ICDAR 2019 Robust Reading Challenge).

Each receipt has three files with the same name:

* `img/<id>.jpg`: the scan.
* `box/<id>.csv`: one ground-truth text line per row, as
  `x1,y1,x2,y2,x3,y3,x4,y4,text` (four corners, then the text, which may contain commas).
* `key/<id>.json`: the key fields `company`, `date`, `address` and `total`.

The annotations have some noise of their own (for example "SDN BND" in a box file
where the key file says "SDN BHD"), so a perfect OCR would not score exactly zero.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .layout import Box, boxes_to_text

FIELDS = ("company", "date", "address", "total")


@dataclass(frozen=True)
class Receipt:
    id: str
    image_path: Path
    lines: tuple[Box, ...]
    fields: dict[str, str]

    @property
    def reference_text(self) -> str:
        """Ground-truth text in the same reading order used for OCR output."""
        return boxes_to_text(list(self.lines))


def parse_box_file(path: Path) -> list[Box]:
    boxes = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split(",", 8)
        if len(parts) < 9:
            continue
        try:
            coords = [int(float(v)) for v in parts[:8]]
        except ValueError:
            continue
        text = parts[8].strip()
        if not text:
            continue
        xs, ys = coords[0::2], coords[1::2]
        boxes.append(Box(text, min(xs), min(ys), max(xs), max(ys)))
    return boxes


def parse_key_file(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return {k: str(data.get(k, "")).strip() for k in FIELDS}


def receipt_ids(root: Path) -> list[str]:
    """Ids that have an image, a box file and a key file."""
    root = Path(root)
    return sorted(
        p.stem
        for p in (root / "img").glob("*.jpg")
        if (root / "box" / f"{p.stem}.csv").exists() and (root / "key" / f"{p.stem}.json").exists()
    )


def load_receipts(root: Path, limit: int | None = None, seed: int = 0) -> list[Receipt]:
    """Load receipts from a folder with `img`, `box` and `key` subfolders.

    With `limit`, a random but reproducible sample is taken (fixed `seed`), so every
    experiment compares configurations on exactly the same receipts.
    """
    root = Path(root)
    ids = receipt_ids(root)
    if limit is not None and limit < len(ids):
        rng = random.Random(seed)
        ids = sorted(rng.sample(ids, limit))
    return [
        Receipt(
            id=i,
            image_path=root / "img" / f"{i}.jpg",
            lines=tuple(parse_box_file(root / "box" / f"{i}.csv")),
            fields=parse_key_file(root / "key" / f"{i}.json"),
        )
        for i in ids
    ]
