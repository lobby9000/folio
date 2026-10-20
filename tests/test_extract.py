"""The extraction rules, one behaviour per test, on invented receipt lines."""

import datetime as dt
from decimal import Decimal

import pytest

from folio.extract import (
    baseline_date,
    baseline_total,
    cash_minus_change,
    extract_date,
    extract_total,
    is_total_line,
)


def total(lines):
    got = extract_total(lines)
    return None if got is None else (got.value, got.rule)


@pytest.mark.parametrize(
    "line",
    ["TOTAL 35.00", "Tota] (Inclusive of GST): 31.02", "Fotal RM Incl. of GST 3.47", "Nett Total: $8.50",
     "Total Amt Payable : 159.00", "iotel After Rounding 85.10", "Qty: 1 TotalRM : 11.60",
     "TTL ATF RND 17.20"],
)
def test_total_lines(line):
    assert is_total_line(line)


@pytest.mark.parametrize(
    "line",
    ["SUB-TOTAL 27.27", "Total Qty 3", "TotalQty 470", "Total Items : 4", "Total (Excluding GST): 26.60",
     "Total GST : 1.68", "TAX TOTAL: 1.44", "Total Savings 2.00", "Total 6% supplies (Inc. GST): 15.26",
     "TOTAL INCLUDES 6% GST 1.66", "Description Qty U.price Total TAX", "Change Due : 0.10"],
)
def test_lines_that_only_look_like_the_total(line):
    assert not is_total_line(line)


def test_last_total_line_wins_over_the_pre_rounding_total():
    lines = ["Total RM Incl. of GST 20.98", "Rounding Adj 0. 02", "Total RM 21.00", "Cash 50.00",
             "CHANGE 29.00"]
    got = extract_total(lines)
    assert (got.value, got.rule, got.line, got.checked) == (Decimal("21.00"), "last total line", 2, True)


def test_reading_stops_at_payment_and_at_the_tax_summary():
    assert total(["TOTAL 35.00", "CASH 50.00", "CHANGE 15.00", "TOTAL 33.02 1.98"])[0] == Decimal("35.00")
    lines = ["Total Sales (Inclusive of GST) : 13.25", "GST Summary Amount Tax", "Total : 12.50 0.75"]
    assert total(lines)[0] == Decimal("13.25")


def test_rounding_adjustment_is_applied_with_its_sign():
    lines = ["Total (Inclusive of GST): 31.02", "Rounding Adj: -0,02", "CASH : 31.00"]
    assert total(lines) == (Decimal("31.00"), "total plus rounding")


def test_rounding_sign_is_inferred_from_the_5_sen_rule():
    assert total(["Total Incl GST 14.64", "Rounding Adj 0.01"])[0] == Decimal("14.65")
    assert total(["Total Incl GST 31.02", "Rounding 0.02"])[0] == Decimal("31.00")


def test_rounded_total_printed_on_the_rounding_line():
    lines = ["TOTAL SALES (INCLUSIVE GST) RM 27.28", "ROUNDING ADJUSTMENT RM", "ROUNDING RM 27.30",
             "CASH RM 27.30"]
    assert total(lines) == (Decimal("27.30"), "total plus rounding")


def test_a_total_that_is_already_round_is_left_alone():
    assert total(["TOTAL 21.00", "Rounding Adj 0.02"]) == (Decimal("21.00"), "last total line")


def test_cash_minus_change_overrides_a_total_it_contradicts_when_the_block_prints_it():
    lines = ["TOTAL AMT RM 60.01", "ROUNDING ADJ 0.01", "RM 60.30", "CASH RM 70.30", "CHANGE RM 10.00"]
    got = extract_total(lines)
    assert (got.value, got.checked) == (Decimal("60.30"), True)
    assert got.rule == "total confirmed by cash minus change"


def test_a_disagreeing_cash_line_alone_does_not_override():
    got = extract_total(["TOTAL 35.00", "CASH 99.00", "Change Returned 20.00"])
    assert (got.value, got.checked) == (Decimal("35.00"), False)


def test_fallbacks_in_order():
    assert total(["Total Sales (Inclusive of GST) :", "29.68", "Thank you"]) == (
        Decimal("29.68"), "value on the next row")
    assert total(["SUBTOTAL :", "CASH 100.00", "CASH CHANGE 55.10"]) == (
        Decimal("44.90"), "cash minus change")
    assert total(["SUB-TOTAL 74.04", "DEBIT CARD 78.10"]) == (Decimal("78.10"), "card payment")
    assert extract_total(["THANK YOU", "PLEASE COME AGAIN"]) is None


def test_cash_minus_change_needs_both_lines():
    assert cash_minus_change(["CASH 50.00", "CHANGE 12.50"]) == Decimal("37.50")
    assert cash_minus_change(["CASH 50.00"]) is None
    assert cash_minus_change(["CASH 5.00", "CHANGE 12.50"]) is None  # misread: change above cash


def test_date_after_a_label_beats_an_earlier_date():
    got = extract_date(["Ref 12/03/2017 0042", "Date: 14/03/2018 10:22"])
    assert (got.value, got.rule, got.line) == (dt.date(2018, 3, 14), "date after a label", 1)


def test_expiry_dates_are_skipped_and_the_first_date_is_used():
    got = extract_date(["Card Expiry: 31/05/18", "05/03/18 15:02 Slip No.: 1010104759"])
    assert (got.value, got.rule) == (dt.date(2018, 3, 5), "first date")
    assert extract_date(["no date here"]) is None


def test_baselines():
    lines = ["Date 01/02/2018", "TOTAL 35.00", "CASH 50.00", "Printed 02/02/2018"]
    assert baseline_total(lines).value == Decimal("50.00")
    assert baseline_date(lines).value == dt.date(2018, 2, 1)
    assert baseline_total(["no amounts"]) is None
