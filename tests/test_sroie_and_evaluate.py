import pytest

from folio.evaluate import Row, compare, exact_mcnemar, render_markdown, summarize
from folio.sroie import load_receipts, parse_box_file, receipt_ids


def test_box_file_keeps_commas_inside_the_text(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("1,2,30,2,30,20,1,20,NO.53 55,57 & 59, JALAN SAGU 18,\n", encoding="utf-8")
    [box] = parse_box_file(path)
    assert box.text == "NO.53 55,57 & 59, JALAN SAGU 18,"
    assert (box.left, box.top, box.right, box.bottom) == (1, 2, 30, 20)


def test_box_file_skips_malformed_lines(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("not,a,box\n1,2,3,4,5,6,7,8,  \n1,2,30,2,30,20,1,20,OK\n", encoding="utf-8")
    assert [b.text for b in parse_box_file(path)] == ["OK"]


def test_load_receipts_reads_text_and_fields(fake_sroie):
    receipts = load_receipts(fake_sroie)
    assert [r.id for r in receipts] == ["r1", "r2"]
    assert receipts[0].reference_text == "FOLIO STORE SDN BHD\nTOTAL 12.25"
    assert receipts[1].fields["total"] == "9.00"


def test_sampling_is_reproducible(fake_sroie):
    a = [r.id for r in load_receipts(fake_sroie, limit=1, seed=3)]
    b = [r.id for r in load_receipts(fake_sroie, limit=1, seed=3)]
    assert a == b and set(a) <= set(receipt_ids(fake_sroie))


def test_exact_mcnemar_known_values():
    assert exact_mcnemar(0, 0) == 1
    assert exact_mcnemar(0, 6) == pytest.approx(2 * 0.5**6)
    assert exact_mcnemar(3, 3) == 1


def _row(receipt, pipeline, cer, total_found):
    return Row(receipt, pipeline, 4, cer, cer, 1 - cer, total_found, True, 90.0, 1.0)


def test_summary_and_paired_comparison():
    rows = [
        _row("a", "new", 0.10, True), _row("a", "old", 0.20, False),
        _row("b", "new", 0.30, True), _row("b", "old", 0.25, True),
        _row("c", "new", 0.05, True), _row("c", "old", 0.15, False),
    ]
    summaries = summarize(rows)
    assert [s.pipeline for s in summaries] == ["new", "old"]
    assert summaries[0].total_found == 3 and summaries[1].total_found == 1

    c = compare(rows, ("new", 4), ("old", 4))
    assert c.n == 3
    assert c.cer_diff_mean == pytest.approx((-0.10 + 0.05 - 0.10) / 3)
    assert (c.a_better, c.b_better) == (2, 1)
    assert (c.total_only_a, c.total_only_b) == (2, 0)

    report = render_markdown(summaries, c, "# Report")
    assert report.startswith("# Report") and "| new | 4 |" in report


def test_unlabelled_fields_are_left_out_of_the_rate():
    rows = [_row("a", "new", 0.1, True), _row("b", "new", 0.2, None), _row("c", "new", 0.3, False)]
    [s] = summarize(rows)
    assert (s.total_found, s.total_n) == (1, 2)
