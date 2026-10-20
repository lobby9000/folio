"""The LLM extractor without a real model: the evidence check, the cache, the cascade and
both backends against fake servers."""

import datetime as dt
import json
import threading
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from folio.evaluate_llm import evaluate, render_markdown
from folio.extract import Extraction
from folio.learn import TotalRanker
from folio.llm import (
    SCHEMA,
    VISION_SCHEMA,
    Answer,
    AnthropicBackend,
    CachedBackend,
    OllamaBackend,
    build_prompt,
    cascade,
    check_answer,
    check_field,
    check_image_answer,
    consult_image,
    consult_text,
)
from folio.ocr import OcrResult
from folio.sroie import Receipt

from .test_learn import labelled_shops, receipt_rows

LINES = ["FOLIO MART", "Date: 19/10/2026", "SUB TOTAL 12.27", "TOTAL RM 12.25", "CASH 20.00", "CHANGE 7.75"]


def answer(total="12.25", total_row=4, date="19/10/2026", date_row=2) -> Answer:
    return Answer({"total": total, "total_row": total_row, "date": date, "date_row": date_row}, 0.5, 300, 20)


def test_prompt_numbers_the_rows_from_one():
    prompt = build_prompt(LINES)
    assert "1: FOLIO MART" in prompt and "4: TOTAL RM 12.25" in prompt and "total" in prompt


@pytest.mark.parametrize(
    ("value", "row", "status", "expected"),
    [
        ("12.25", 4, "grounded", Decimal("12.25")),
        ("RM 12.25", 4, "grounded", Decimal("12.25")),  # compared as values, not strings
        ("12.25", 3, "wrong row", Decimal("12.25")),  # printed, but not on the row it cited
        ("31.45", 4, "not in the text", None),  # invented: rejected
        ("twelve", 4, "unreadable", None),
        ("", 0, "no answer", None),
    ],
)
def test_a_total_is_kept_only_if_it_is_printed(value, row, status, expected):
    got = check_field("total", value, row, LINES)
    assert (got.status, got.value) == (status, expected)


def test_dates_are_checked_the_same_way():
    got = check_answer(answer(date="2026-10-19", date_row=2).data, LINES)["date"]
    assert (got.status, got.value, got.row) == ("grounded", dt.date(2026, 10, 19), 1)
    assert check_answer({"total": "12.25"}, LINES)["date"].status == "no answer"  # missing keys tolerated


class FakeBackend:
    name, model = "fake", "fake-1"

    def __init__(self, reply: Answer):
        self.reply, self.prompts = reply, []

    def ask(self, prompt: str) -> Answer:
        self.prompts.append(prompt)
        return self.reply


def test_cache_answers_once_per_receipt_and_prompt(tmp_path):
    backend = FakeBackend(answer())
    cached = CachedBackend(backend, tmp_path)
    first = cached.ask("r1", build_prompt(LINES))
    again = cached.ask("r1", build_prompt(LINES))
    changed = cached.ask("r1", build_prompt(LINES + ["THANK YOU"]))
    assert first.data == again.data == changed.data and cached.calls == 2 and len(backend.prompts) == 2
    assert (tmp_path / "fake__fake-1" / "r1.json").exists()


def ranked(value: str, score: float) -> Extraction:
    return Extraction(Decimal(value), 3, LINES[3], "logistic ranker", None, score)


def rules(value: str, checked: bool | None) -> Extraction:
    return Extraction(Decimal(value), 3, LINES[3], "last total line", checked)


def test_cascade_trusts_the_cash_check_and_a_confident_ranker():
    never = FakeBackend(answer())

    def consult():
        return consult_text(never.ask, LINES)

    date = Extraction(dt.date(2026, 10, 19), 1, LINES[1], "date after a label")
    got = cascade(LINES, rules("12.25", True), date, ranked("12.27", 0.5), consult)
    assert (got.route, got.total.value, never.prompts) == ("confirmed by cash", Decimal("12.25"), [])
    got = cascade(LINES, rules("12.27", False), date, ranked("12.25", 0.99), consult)
    assert (got.route, got.total.value, never.prompts) == ("confident ranker", Decimal("12.25"), [])


def test_cascade_asks_when_unsure_and_keeps_only_grounded_answers():
    date = Extraction(dt.date(2026, 10, 19), 1, LINES[1], "date after a label")
    got = cascade(LINES, rules("12.27", False), date, ranked("12.27", 0.4),
                  lambda: consult_text(FakeBackend(answer()).ask, LINES))
    assert (got.route, got.total.value, got.total.rule, got.total.checked) == (
        "llm", Decimal("12.25"), "llm (grounded)", True)
    assert got.date.value == date.value and got.date.rule == "llm (grounded)"  # asked anyway: its date wins
    invented = cascade(LINES, None, None, ranked("12.27", 0.4),
                       lambda: consult_text(FakeBackend(answer(total="31.45")).ask, LINES))
    assert invented.total.value == Decimal("12.27") and invented.llm["total"].status == "not in the text"
    assert invented.date.value == dt.date(2026, 10, 19) and invented.date.rule == "llm (grounded)"


def test_anthropic_backend_sends_the_schema_through_the_real_sdk():
    anthropic = pytest.importorskip("anthropic")
    try:  # recent SDKs are built on httpx2, older ones on httpx
        import httpx2 as httpx
    except ImportError:
        httpx = pytest.importorskip("httpx")
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        body = {"id": "msg_1", "type": "message", "role": "assistant", "model": seen["model"],
                "content": [{"type": "text", "text": json.dumps(answer().data)}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 321, "output_tokens": 24}}
        return httpx.Response(200, json=body)

    client = anthropic.Anthropic(api_key="test-key", base_url="http://anthropic.test", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    got = AnthropicBackend(client=client).ask(build_prompt(LINES))
    assert got.data == answer().data and (got.input_tokens, got.output_tokens) == (321, 24)
    assert seen["model"] == "claude-haiku-5-5" and seen["system"].startswith("You read OCR text")
    assert seen["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert seen["thinking"] == {"type": "disabled"}
    assert "4: TOTAL RM 12.25" in seen["messages"][0]["content"]


def test_ollama_backend_speaks_the_chat_api():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (http.server naming)
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = {"message": {"role": "assistant", "content": json.dumps(answer().data)},
                     "prompt_eval_count": 410, "eval_count": 31}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        backend = OllamaBackend(model="qwen2.5:3b", host=f"127.0.0.1:{server.server_port}")
        got = backend.ask(build_prompt(LINES))
    finally:
        server.shutdown()
    assert got.data == answer().data and (got.input_tokens, got.output_tokens) == (410, 31)
    assert seen["path"] == "/api/chat" and seen["body"]["format"] == SCHEMA
    assert seen["body"]["model"] == "qwen2.5:3b" and seen["body"]["stream"] is False


def test_evaluation_reports_routes_and_the_model_alone():
    ranker = TotalRanker.fit(labelled_shops(30), "logistic")
    rows = receipt_rows([("FOLIO MART", None), ("TOTAL", "9.95"), ("THANK YOU", None)])
    receipt = Receipt("r1", Path("r1.jpg"), (), {"company": "", "date": "", "address": "", "total": "9.95"})
    ocr = OcrResult([b for row in rows for b in row])
    calls = []

    def ask(rid, prompt, image=None, schema=None):
        calls.append(rid)
        return answer(total="9.95", total_row=2, date="", date_row=0)

    cascade_rows = evaluate([(receipt, ocr)], ranker, ask, mode="cascade", threshold=1.01)
    all_rows = evaluate([(receipt, ocr)], ranker, ask, mode="all", threshold=0.0)
    assert calls == ["r1", "r1"]
    assert cascade_rows[0].route == "llm" and cascade_rows[0].final_total_ok and cascade_rows[0].llm_total_ok
    assert all_rows[0].route == "llm for the date" and all_rows[0].asked  # no date on this receipt
    md = render_markdown(all_rows, "header")
    assert "| LLM alone (grounded answers only) | 1 of 1" in md and "## Cost" in md


def test_list_price_cost_and_new_request_accounting(tmp_path):
    from folio.llm import cost_usd

    assert cost_usd("claude-haiku-5-5", 1_000_000, 0) == pytest.approx(0.10)
    assert cost_usd("claude-sonnet-5-5", 0, 100_000) == pytest.approx(1.00)
    assert cost_usd("qwen2.5:3b", 1000, 1000) is None
    cached = CachedBackend(FakeBackend(answer()), tmp_path)
    cached.ask("r1", "a prompt")
    cached.ask("r1", "a prompt")  # from the cache: free
    assert (cached.calls, cached.new_input_tokens, cached.new_output_tokens) == (1, 300, 20)


def test_anthropic_thinking_options():
    class Client:
        def __init__(self):
            self.messages = self
            self.kwargs = None

        def create(self, **kwargs):
            from types import SimpleNamespace

            self.kwargs = kwargs
            text = SimpleNamespace(type="text", text=json.dumps(answer().data))
            blocks = [SimpleNamespace(type="thinking"), text]
            return SimpleNamespace(content=blocks, stop_reason="end_turn", usage=None)

    client = Client()
    backend = AnthropicBackend(client=client, thinking="low")
    assert backend.ask("p").data == answer().data and backend.label == "claude-haiku-5-5-thinking-low"
    body = client.kwargs["extra_body"]
    assert body["thinking"] == {"type": "adaptive"} and body["output_config"]["effort"] == "low"
    assert client.kwargs["max_tokens"] == 8000
    with pytest.raises(ValueError):
        AnthropicBackend(client=client, thinking="max")


class ImageBackend:
    name, model, label = "fake", "fake-vision", "fake-vision"

    def __init__(self, reply: dict):
        self.reply, self.seen = reply, []

    def ask(self, prompt, image=None, schema=None):
        self.seen.append((prompt, image, schema))
        return Answer(self.reply, 0.4, 900, 30)


def test_reading_the_scan_keeps_values_ocr_misread():
    lines = ["FOLIO MART", "Total Ri 3145", "Cash 50.00"]  # OCR lost the decimal point
    reply = {"total": "RM 31.45", "total_line": "Total RM 31.45",
             "date": "19/10/2026", "date_line": "19/10/2026 10:22"}
    backend = ImageBackend(reply)
    answer, checked = consult_image(backend.ask, b"\xff\xd8fake-jpeg", lines)
    assert backend.seen[0][1] == b"\xff\xd8fake-jpeg" and backend.seen[0][2] == VISION_SCHEMA
    assert (checked["total"].status, checked["total"].value, checked["total"].row) == (
        "read from the image", Decimal("31.45"), None)
    got = cascade(lines, None, None, ranked("50.00", 0.3), lambda: (answer, checked))
    assert got.total.value == Decimal("31.45") and got.total.line == -1
    assert got.total.evidence == "Total RM 31.45"
    assert check_image_answer({"total": "", "date": "x"}, lines)["date"].status == "unreadable"


def test_image_and_text_answers_are_cached_apart(tmp_path):
    empty = {"total": "1.00", "total_line": "", "date": "", "date_line": ""}
    cached = CachedBackend(ImageBackend(empty), tmp_path)
    cached.ask("r1", "prompt", b"img", VISION_SCHEMA)
    cached.ask("r1", "prompt", b"img", VISION_SCHEMA)
    cached.ask("r1", "prompt", b"other image", VISION_SCHEMA)
    assert cached.calls == 2 and (tmp_path / "fake__fake-vision" / "r1.image.json").exists()


def test_anthropic_backend_sends_the_image_as_base64():
    class Client:
        def __init__(self):
            self.messages, self.kwargs = self, None

        def create(self, **kwargs):
            from types import SimpleNamespace

            self.kwargs = kwargs
            text = SimpleNamespace(type="text", text=json.dumps({"total": "", "total_line": "", "date": "",
                                                                 "date_line": ""}))
            return SimpleNamespace(content=[text], stop_reason="end_turn", usage=None)

    client = Client()
    AnthropicBackend(client=client).ask("read it", b"\x89PNG\r\n\x1a\nrest", VISION_SCHEMA)
    content = client.kwargs["messages"][0]["content"]
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[1] == {"type": "text", "text": "read it"}
    assert client.kwargs["extra_body"]["output_config"]["format"]["schema"] == VISION_SCHEMA


def test_a_trusted_total_without_a_date_still_asks_for_the_date():
    backend = FakeBackend(answer(total="99.99"))  # its total is ignored: the cash check already settled it
    got = cascade(LINES, rules("12.25", True), None, ranked("12.27", 0.5),
                  lambda: consult_text(backend.ask, LINES))
    assert got.route == "llm for the date" and got.total.value == Decimal("12.25")
    assert got.date.value == dt.date(2026, 10, 19) and len(backend.prompts) == 1
