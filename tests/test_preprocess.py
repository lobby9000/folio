import numpy as np
import pytest

from folio.preprocess import (
    PIPELINES,
    binarize_adaptive,
    binarize_otsu,
    deskew,
    estimate_skew,
    rotate,
    to_gray,
    upscale,
)


def test_to_gray_accepts_gray_colour_and_alpha(page):
    colour = np.dstack([page] * 3)
    alpha = np.dstack([page] * 3 + [np.full_like(page, 255)])
    for img in (page, colour, alpha):
        gray = to_gray(img)
        assert gray.ndim == 2 and gray.dtype == np.uint8


@pytest.mark.parametrize("binarize", [binarize_otsu, binarize_adaptive])
def test_binarizers_return_pure_black_and_white(page, binarize):
    out = binarize(page)
    assert set(np.unique(out)) <= {0, 255}


@pytest.mark.parametrize("angle", [-6.0, -2.5, 3.0, 7.5])
def test_estimate_skew_recovers_a_known_rotation(page, angle):
    tilted = rotate(page, angle)
    assert estimate_skew(tilted) == pytest.approx(-angle, abs=0.3)


def test_deskew_leaves_a_straight_page_alone(page):
    assert deskew(page) is page


def test_rotation_keeps_the_whole_page(page):
    ink_before = int((page < 128).sum())
    ink_after = int((rotate(page, 8.0) < 128).sum())
    assert ink_after == pytest.approx(ink_before, rel=0.1)


def test_upscale_doubles_both_sides(page):
    assert upscale(page, 2.0).shape == (2 * page.shape[0], 2 * page.shape[1])


@pytest.mark.parametrize("name", sorted(PIPELINES))
def test_every_pipeline_returns_a_gray_8bit_image(page, name):
    out = PIPELINES[name](np.dstack([page] * 3))
    assert out.ndim == 2 and out.dtype == np.uint8
