"""Amounts and dates as they come out of receipts and out of OCR."""

import datetime as dt
from decimal import Decimal

import pytest

from folio.normalize import find_amounts, find_dates, parse_amount, parse_date


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("TOTAL RM 60.01", "60.01"),
        ("Total Incl. GST@6% RM 5,90", "5.90"),  # comma as decimal separator
        ("Total RN 85. 54", "85.54"),  # stray space after the separator
        ("Total (Excluding GST): 26 .60", "26.60"),  # and before it
        ("NET TOTAL 174.9", "174.90"),  # dropped trailing zero
        ("ROUNDING ADJUSTMENT RM .02", "0.02"),  # missing leading zero
        ("Change : RMO.25", "0.25"),  # currency glued on, O read for 0
        ("Total Incl. GST@6% <RM_34.80", "34.80"),
        ("Total Amount: $8.90", "8.90"),
        ("GRAND TOTAL 1,234.50", "1234.50"),  # thousands separator
    ],
)
def test_amounts_survive_receipt_and_ocr_quirks(text, value):
    assert [a.value for a in find_amounts(text)] == [Decimal(value)]


@pytest.mark.parametrize(
    "text",
    [
        "TOTAL QTY 3",  # integers are counts, not money
        "CHANGE O. OO",  # no real digit at all
        "TotalQty 470 465.340",  # three decimals: not a price, probably a misread
        "GST @6.00%",  # a rate
        "0.C. WHITE 2.13.12",  # a code
        "Total Sales Incl GST 5/90",  # slash for a decimal point is left as an OCR error
    ],
)
def test_things_that_are_not_amounts(text):
    assert find_amounts(text) == []


def test_amount_positions_point_back_into_the_line():
    line = "Rounding Adj: -0,02  Total 31.00"
    found = find_amounts(line)
    assert [line[a.start:a.end] for a in found] == ["0,02", "31.00"]


def test_labels_parse_with_or_without_currency():
    assert parse_amount("RM 34.80") == Decimal("34.80")
    assert parse_amount("$8.90") == Decimal("8.90")
    assert parse_amount("9") == Decimal("9")
    assert parse_amount("") is None


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("19/10/2018 20:49:59 #01", (2018, 10, 19)),
        ("Date: 14-03-2018 10:22:59", (2018, 3, 14)),
        ("26/02/18 08:28 10886", (2018, 2, 26)),  # two-digit year
        ("11.02.18 14:08 2007", (2018, 2, 11)),
        ("fate : 4/02/2017 03:30:27PH", (2017, 2, 4)),
        ("O7/01/18 14:41 Slip", (2018, 1, 7)),  # O read for 0
        ("Closed by - c001- 07/04/2018-18:11:05", (2018, 4, 7)),  # time glued on
        ("2018-04-06 17:36:35", (2018, 4, 6)),
        ("10 Mar 2018 18:24", (2018, 3, 10)),
        ("05-Aug-2017", (2017, 8, 5)),
        ("25/FEB/2017", (2017, 2, 25)),
        ("Print Time : Monday, 11 December, 2017 8:47:34 PM", (2017, 12, 11)),
        ("15 Jun 18 08:34:31 PM", (2018, 6, 15)),
        ("OCT 3, 2016", (2016, 10, 3)),
    ],
)
def test_date_formats(text, value):
    assert [d.value for d in find_dates(text)] == [dt.date(*value)]


def test_day_first_unless_impossible_with_slashes():
    assert parse_date("03/04/2018") == dt.date(2018, 4, 3)
    assert parse_date("12/25/2018") == dt.date(2018, 12, 25)
    assert find_dates("2.13.12") == []  # no month-first rescue for dots or dashes


@pytest.mark.parametrize("text", ["9018-04-06 17:36:35", "Date — 04-04-2048 14:98:96",
                                  "TAXINVOICE 92/12/2017", "BO19 MARIGOLD HL MILK", "Date: 23.116/2017"])
def test_impossible_or_implausible_dates_are_rejected(text):
    assert find_dates(text) == []
