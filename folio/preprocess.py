"""Image preprocessing before OCR.

Every step is a pure function from a NumPy image to a NumPy image, so steps can be
combined freely and tested on their own. `PIPELINES` names the combinations that
`folio eval-ocr` compares.

Tesseract expects dark text on a light background, in horizontal lines, scanned at
300 dpi or more. Each step below moves a scan closer to that.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

Image = np.ndarray


def to_gray(img: Image) -> Image:
    """Return a single-channel 8-bit image. Colour rarely helps a text reader."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: Image, ksize: int = 3) -> Image:
    """Median filter: removes isolated specks while keeping letter edges sharp."""
    return cv2.medianBlur(gray, ksize)


def binarize_otsu(gray: Image) -> Image:
    """Black and white with one global threshold chosen by Otsu's method.

    Otsu picks the threshold that minimises the variance inside the two resulting
    classes (ink and paper). It works well on flat, evenly lit scans.
    """
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def binarize_adaptive(gray: Image, block_size: int = 31, c: int = 15) -> Image:
    """Black and white with a threshold computed for each neighbourhood.

    Copes with shadows and uneven lighting, which a single global threshold cannot.
    `block_size` is the neighbourhood side in pixels and `c` is subtracted from the
    local weighted mean, so faint background texture stays white.
    """
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def upscale(gray: Image, factor: float = 2.0) -> Image:
    """Enlarge with cubic interpolation, so small print reaches a size Tesseract reads well."""
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def rotate(img: Image, angle: float, border: int = 255) -> Image:
    """Rotate counter-clockwise by `angle` degrees, enlarging the canvas so no corner is cut."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    m[0, 2] += new_w / 2 - w / 2
    m[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(
        img, m, (new_w, new_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    )


def _shrink(gray: Image, max_side: int) -> Image:
    scale = max_side / max(gray.shape[:2])
    if scale >= 1:
        return gray
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _profile_score(ink: Image, angle: float) -> float:
    """Variance of the row sums after rotating: high when text lines are horizontal."""
    rotated = rotate(ink, angle, border=0)
    return float(np.var(rotated.sum(axis=1, dtype=np.float64)))


def estimate_skew(gray: Image, max_angle: float = 10.0, step: float = 0.5) -> float:
    """Angle in degrees that makes the text lines horizontal (projection-profile method).

    With ink as white pixels, horizontal lines of text give a row profile with sharp
    peaks (lines) and valleys (gaps between lines), so the right angle maximises the
    variance of the row sums. A coarse search over [-max_angle, max_angle] is refined
    around the best angle in steps of 0.1 degrees. The search runs on a shrunken copy,
    because the angle does not depend on resolution.
    """
    small = _shrink(gray, max_side=800)
    ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coarse = np.arange(-max_angle, max_angle + 1e-9, step)
    best = max(coarse, key=lambda a: _profile_score(ink, float(a)))
    fine = np.arange(best - step, best + step + 1e-9, 0.1)
    best = max(fine, key=lambda a: _profile_score(ink, float(a)))
    return round(float(best), 2)


def deskew(gray: Image, max_angle: float = 10.0) -> Image:
    """Straighten the page. Small angles (under 0.3 degrees) are left alone."""
    angle = estimate_skew(gray, max_angle=max_angle)
    if abs(angle) < 0.3:
        return gray
    return rotate(gray, angle, border=255)


def _gray(img: Image) -> Image:
    return to_gray(img)


def _otsu(img: Image) -> Image:
    return binarize_otsu(to_gray(img))


def _adaptive(img: Image) -> Image:
    return binarize_adaptive(denoise(to_gray(img)))


def _deskew_otsu(img: Image) -> Image:
    return binarize_otsu(deskew(to_gray(img)))


def _gray_x2(img: Image) -> Image:
    return upscale(to_gray(img), 2.0)


PIPELINES: dict[str, Callable[[Image], Image]] = {
    "gray": _gray,
    "gray-x2": _gray_x2,
    "otsu": _otsu,
    "adaptive": _adaptive,
    "deskew+otsu": _deskew_otsu,
}
"""Named preprocessing pipelines, from the least processing to the most."""
