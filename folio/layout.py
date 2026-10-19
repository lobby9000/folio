"""Text boxes and reading order.

OCR engines and ground-truth annotations both describe a page as boxes with text,
but they split it differently: Tesseract returns one box per word, while SROIE
annotates whole lines, sometimes several on the same row. To compare the two
fairly, both go through the same rule: group boxes into rows from top to bottom and
read each row from left to right. That way the character error rate measures
reading, not disagreements about reading order.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """A piece of text and its axis-aligned bounding box, in pixels."""

    text: str
    left: float
    top: float
    right: float
    bottom: float
    conf: float | None = None  # OCR confidence from 0 to 100, None for ground truth

    @property
    def height(self) -> float:
        return max(self.bottom - self.top, 1.0)

    @property
    def y_center(self) -> float:
        return (self.top + self.bottom) / 2


def group_rows(boxes: list[Box], tolerance: float = 0.5) -> list[list[Box]]:
    """Group boxes into rows, top to bottom, each row sorted left to right.

    A box joins the current row when its vertical centre is within `tolerance`
    times the median box height of the row's running mean centre.
    """
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: (b.y_center, b.left))
    median_height = statistics.median(b.height for b in ordered)
    rows: list[list[Box]] = []
    current = [ordered[0]]
    row_y = ordered[0].y_center
    for box in ordered[1:]:
        if abs(box.y_center - row_y) <= tolerance * median_height:
            current.append(box)
            row_y = sum(b.y_center for b in current) / len(current)
        else:
            rows.append(sorted(current, key=lambda b: b.left))
            current, row_y = [box], box.y_center
    rows.append(sorted(current, key=lambda b: b.left))
    return rows


def rows_to_text(rows: list[list[Box]]) -> str:
    """One line per row, words separated by single spaces."""
    return "\n".join(" ".join(b.text for b in row) for row in rows)


def boxes_to_text(boxes: list[Box]) -> str:
    """Shortcut for `rows_to_text(group_rows(boxes))`."""
    return rows_to_text(group_rows(boxes))
