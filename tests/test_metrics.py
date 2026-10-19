import random

import pytest

from folio.metrics import bag_of_words, cer, field_found, normalize, wer, wilson_interval


def textbook_levenshtein(a: str, b: str) -> int:
    """The dynamic-programming definition, used to check the fast library."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def test_cer_matches_the_textbook_edit_distance():
    rng = random.Random(0)
    checked = 0
    for _ in range(300):
        a = "".join(rng.choice("ab c1.") for _ in range(rng.randint(1, 30)))
        b = "".join(rng.choice("ab c1.") for _ in range(rng.randint(0, 30)))
        na, nb = normalize(a), normalize(b)
        if not na:
            continue
        assert cer(a, b) == pytest.approx(textbook_levenshtein(na, nb) / len(na))
        checked += 1
    assert checked > 200


def test_cer_counts_each_kind_of_edit():
    assert cer("TOTAL 9.00", "TOTAL 9.00") == 0
    assert cer("TOTAL 9.00", "TOTAL 9.0") == pytest.approx(1 / 10)  # deletion
    assert cer("TOTAL 9.00", "TOTAL 9.000") == pytest.approx(1 / 10)  # insertion
    assert cer("TOTAL 9.00", "T0TAL 9.00") == pytest.approx(1 / 10)  # substitution


def test_cer_can_exceed_one_when_text_is_invented():
    assert cer("A", "ABCDE") == 4


def test_case_and_spacing_are_ignored():
    assert cer("Total  9.00\n", "TOTAL 9.00") == 0


def test_wer_counts_whole_words():
    assert wer("total 9.00 cash", "total 900 cash") == pytest.approx(1 / 3)


def test_bag_of_words_ignores_order():
    prf = bag_of_words("a b c d", "d c b a")
    assert prf.precision == prf.recall == prf.f1 == 1


def test_bag_of_words_partial_match():
    prf = bag_of_words("a b c d", "a b x")
    assert prf.precision == pytest.approx(2 / 3)
    assert prf.recall == pytest.approx(2 / 4)


def test_field_found_respects_token_boundaries():
    assert field_found("9.00", "TOTAL 9.00 RM")
    assert not field_found("9.00", "TOTAL 19.00")
    assert not field_found("9.00", "TOTAL 9.001")
    assert field_found("25/12/2018", "Date: 25/12/2018 13:39")
    assert not field_found("", "anything")


def test_wilson_interval():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0 and 0 < hi < 0.35
    lo, hi = wilson_interval(81, 100)  # p = 0.81, z = 1.96
    assert lo == pytest.approx(0.7222, abs=1e-3)
    assert hi == pytest.approx(0.8749, abs=1e-3)
