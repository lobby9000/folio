"""Layout clustering on invented receipts from a few shops: merchant keys, representations,
groups and their scores, grouped folds for the ranker, and the stage 4 report."""

import random
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pytest

from folio.cluster import (
    KINDS,
    cluster,
    distances,
    merchant,
    nearest_distance,
    represent,
    scores,
)
from folio.evaluate_cluster import render_markdown, run
from folio.learn import cross_validate, group_folds
from folio.ocr import OcrResult
from folio.sroie import Receipt

from .test_learn import labelled_shops, receipt_rows

SHOPS = {
    "KEDAI BUKU MAJU SDN BHD": ("JALAN AMPANG 12", "STATIONERY AND BOOKS"),
    "RESTORAN SELERA KITA": ("TAMAN MELATI 3", "NASI LEMAK TEH TARIK"),
    "MR. D.I.Y. (KUCHAI) SDN BHD": ("KUCHAI LAMA 7", "HOME IMPROVEMENT HARDWARE"),
    "FARMASI SIHAT": ("BANGSAR SOUTH 9", "PHARMACY HEALTH CARE"),
}


def shop_receipt(name: str, k: int, rng: random.Random) -> tuple[Receipt, OcrResult]:
    address, trade = SHOPS[name]
    total = Decimal(rng.randint(500, 9000)) / 100
    lines = [(name, None), (address, None), (trade, None), ("TAX INVOICE", None)]
    lines += [(f"ITEM {trade.split()[0]} {j}", f"{Decimal(rng.randint(100, 900)) / 100:.2f}")
              for j in range(3)]
    lines += [("TOTAL", f"{total:.2f}"), ("THANK YOU PLEASE COME AGAIN", None)]
    ocr = OcrResult([b for row in receipt_rows(lines) for b in row])
    receipt = Receipt(f"{name[:4]}{k}", Path(f"missing/{k}.jpg"), (),
                      {"company": name, "date": "", "address": address, "total": f"{total:.2f}"})
    return receipt, ocr


def shops(names, per_shop: int, seed: int = 0):
    rng = random.Random(seed)
    return [shop_receipt(name, k, rng) for name in names for k in range(per_shop)]


@pytest.mark.parametrize(
    ("company", "key"),
    [
        ("MR. D.I.Y. (KUCHAI) SDN BHD", "MR D I Y"),
        ("MR D.I.Y. (M) SDN BHD", "MR D I Y"),
        ("AEON CO. (M) BHD", "AEON CO"),
        ("GL HANDICRAFT & TAIL ORING", "GL HANDICRAFT"),
        ("restoran wan sheng", "RESTORAN WAN SHENG"),
        ("SANYU STATIONERY SHOP", "SANYU STATIONERY SHOP"),
        ("YONG SOON FATT S/B", "YONG SOON FATT"),
    ],
)
def test_merchant_keys_merge_chains_and_drop_legal_suffixes(company, key):
    assert merchant(company) == key


def test_text_groups_receipts_by_shop():
    receipts = shops(list(SHOPS), per_shop=4)
    X, fitted = represent("OCR text, words", [r for r, _ in receipts], [o for _, o in receipts])
    groups = cluster(distances(X), 0.7)
    truth = [merchant(r.fields["company"]) for r, _ in receipts]
    s = scores(truth, groups)
    assert s["groups"] == s["merchants"] == 4 and s["ami"] == pytest.approx(1.0)
    assert s["pair_precision"] == s["pair_recall"] == 1.0
    # a new receipt from a known shop lands next to its shop in the fitted space
    new, ocr = shop_receipt("FARMASI SIHAT", 99, random.Random(5))
    near = distances(represent("OCR text, words", [new], [ocr], fitted)[0], X)
    assert truth[int(near.argmin())] == "FARMASI SIHAT"


def test_scores_tell_merged_groups_from_split_ones():
    truth = ["A", "A", "B", "B"]
    one = scores(truth, [0, 0, 0, 0])
    assert one["completeness"] == pytest.approx(1.0) and one["homogeneity"] == pytest.approx(0.0)
    assert one["pair_precision"] == pytest.approx(2 / 6) and one["pair_recall"] == 1.0
    apart = scores(truth, [0, 1, 2, 3])
    assert apart["homogeneity"] == pytest.approx(1.0) and apart["pair_recall"] == 0.0


def test_distances_survive_empty_receipts_and_nearest_skips_itself():
    X = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.1]])
    D = distances(X)
    assert D[1, 0] == pytest.approx(1.0) and D[0, 2] < 0.01
    nearest = nearest_distance(D, exclude_self=True)
    assert nearest[1] == pytest.approx(1.0) and nearest[0] == nearest[2] == pytest.approx(D[0, 2])
    assert cluster(D[:1, :1], 0.5).tolist() == [0]


def test_every_representation_gives_one_row_per_receipt(tmp_path):
    receipts = shops(list(SHOPS)[:2], per_shop=2)
    image = tmp_path / "scan.jpg"
    cv2.imwrite(str(image), np.full((300, 120), 255, dtype=np.uint8))
    with_image = [(Receipt(r.id, image, (), r.fields), o) for r, o in receipts]
    for kind in KINDS:
        X, _ = represent(kind, [r for r, _ in with_image], [o for _, o in with_image])
        assert X.shape[0] == 4
    grid, _ = represent("word positions", [receipts[0][0]], [receipts[0][1]])
    assert grid.shape == (1, 24 * 8) and grid.sum() == len(receipts[0][1].words)
    with pytest.raises(ValueError):
        represent("colour", [], [])


def test_group_folds_keep_groups_whole_and_balance_the_folds():
    groups = ["a"] * 5 + ["b"] * 4 + ["c"] * 3 + ["d"] * 2 + ["e"] * 2 + ["f"]
    fold = group_folds(groups, folds=3, seed=0)
    for g in set(groups):
        assert len({int(f) for f, h in zip(fold, groups, strict=True) if h == g}) == 1
    assert sorted(np.bincount(fold, minlength=3)) == [5, 6, 6]
    assert (group_folds(groups, 3, 0) == fold).all()
    scores_ = cross_validate("logistic", labelled_shops(30), folds=3, groups=[str(k % 6) for k in range(30)])
    assert len(scores_) == 3 and all(0.9 <= s <= 1 for s in scores_)


def test_stage_4_report_on_invented_shops():
    names = list(SHOPS)
    design = shops(names[:3], per_shop=6, seed=1)
    test = shops(names[1:], per_shop=3, seed=2)  # one shop the design receipts never saw
    llm_ok = {r.id: True for r, _ in test}
    report = run(design, test, ranker=None, llm_ok=llm_ok, thresholds=(0.5, 0.7), repeats=1)
    assert report.kind in KINDS and report.threshold in (0.5, 0.7)
    assert report.test_scores["merchants"] == 3 and len(report.rows) == 9
    new = [r for r in report.rows if not r.seen_merchant]
    assert len(new) == 3 and all(r.merchant == "FARMASI SIHAT" for r in new)
    assert all(r.known_layout == r.seen_merchant for r in report.rows)
    assert all(r.llm_ok and r.ranker_ok is None for r in report.rows)
    assert set(report.folds) == {"random receipts (stage 3)", "whole merchants",
                                 "whole groups found without labels"}
    md = render_markdown(report, "header")
    for part in ("## Choosing a representation", "## Test", "Largest groups", "## Known and new layouts",
                 "## Honest folds for the ranker", "| 5 or more |"):
        assert part in md
