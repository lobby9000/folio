from folio.layout import Box, boxes_to_text, group_rows


def test_boxes_on_one_row_are_read_left_to_right_even_if_slightly_offset():
    boxes = [
        Box("9.00", 300, 102, 360, 120),
        Box("TOTAL", 10, 100, 80, 118),
        Box("CASH", 10, 140, 70, 158),
    ]
    assert boxes_to_text(boxes) == "TOTAL 9.00\nCASH"


def test_rows_are_ordered_top_to_bottom():
    boxes = [Box("third", 0, 200, 50, 220), Box("first", 0, 0, 50, 20), Box("second", 0, 100, 50, 120)]
    assert [row[0].text for row in group_rows(boxes)] == ["first", "second", "third"]


def test_line_boxes_and_word_boxes_give_the_same_text():
    # Ground truth annotates lines, OCR returns words: the same reading rule must agree.
    lines = [Box("TOTAL 9.00", 10, 100, 360, 120)]
    words = [Box("TOTAL", 10, 101, 80, 119), Box("9.00", 300, 100, 360, 120)]
    assert boxes_to_text(lines) == boxes_to_text(words)


def test_empty_page():
    assert boxes_to_text([]) == ""
