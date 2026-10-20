"""Ask a language model for the total and the date, with the row that proves each one.

The model sees the receipt's OCR rows, numbered, and must answer through a JSON
schema: for each field, the value as printed and the number of the row it read it
from. Nothing is taken on trust. An answer is kept only if the value, once
normalised, is actually printed in the OCR text, so the model can choose between
amounts on the receipt but cannot invent one. That is what "textual evidence" means
here, and it is what keeps a hallucinated total out of the results.

Two backends speak the same schema:

* `anthropic`: the Claude API with structured outputs. Needs `pip install anthropic`
  and an `ANTHROPIC_API_KEY`. A key that is not scoped to one workspace also needs
  `ANTHROPIC_WORKSPACE_ID` (the `wrkspc_...` ID from the Console's Workspaces page).
* `ollama`: a local model served by Ollama (`ollama pull qwen2.5:3b`), so nothing
  leaves the machine. It uses Ollama's structured outputs on `/api/chat`.

Every answer is cached on disk with the prompt's hash, so an evaluation can be
stopped and resumed, and rerunning it costs nothing.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

from .extract import Extraction, cash_minus_change
from .normalize import find_amounts, find_dates, parse_amount, parse_date

PROMPT_VERSION = 1

SYSTEM = (
    "You read OCR text of shop receipts from Malaysia and extract fields. The OCR has errors: letters "
    "read as digits, missing decimal points, words split or merged. Answer only through the JSON schema."
)

INSTRUCTIONS = (
    "Rows of the receipt, numbered:\n\n{rows}\n\n"
    "Find two fields.\n"
    "total: the final amount the customer paid, including tax and after any rounding. It is usually on a "
    "line such as TOTAL, NETT TOTAL, GRAND TOTAL, TOTAL AMOUNT PAYABLE or TOTAL ROUNDED. It is not a "
    "subtotal, a tax amount, the cash tendered or the change.\n"
    "date: the date of the purchase.\n"
    "For each field give the value exactly as printed and the number of the row it is on. If a field is "
    "not in the text, give an empty string and row 0."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "total": {"type": "string", "description": "the total exactly as printed, or empty"},
        "total_row": {"type": "integer", "description": "row number of the total, or 0"},
        "date": {"type": "string", "description": "the purchase date exactly as printed, or empty"},
        "date_row": {"type": "integer", "description": "row number of the date, or 0"},
    },
    "required": ["total", "total_row", "date", "date_row"],
    "additionalProperties": False,
}


def build_prompt(lines: Sequence[str]) -> str:
    return INSTRUCTIONS.format(rows="\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines)))


# Reading the scan itself instead of the OCR text. The model quotes the line it read
# each value from, since there are no OCR row numbers to cite.
VISION_INSTRUCTIONS = (
    "This is a scanned shop receipt from Malaysia.\n\n"
    "Find two fields.\n"
    "total: the final amount the customer paid, including tax and after any rounding. It is usually on a "
    "line such as TOTAL, NETT TOTAL, GRAND TOTAL, TOTAL AMOUNT PAYABLE or TOTAL ROUNDED. It is not a "
    "subtotal, a tax amount, the cash tendered or the change.\n"
    "date: the date of the purchase.\n"
    "For each field give the value exactly as printed and the whole line it is printed on. If a field is "
    "not on the receipt, give empty strings."
)

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "total": {"type": "string", "description": "the total exactly as printed, or empty"},
        "total_line": {"type": "string", "description": "the whole line the total is printed on"},
        "date": {"type": "string", "description": "the purchase date exactly as printed, or empty"},
        "date_line": {"type": "string", "description": "the whole line the date is printed on"},
    },
    "required": ["total", "total_line", "date", "date_line"],
    "additionalProperties": False,
}


def _media_type(image: bytes) -> str:
    return "image/png" if image[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"


@dataclass
class Answer:
    data: dict
    seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class AnthropicBackend:
    """Claude through the Messages API with structured outputs.

    `thinking` is "off" (no reasoning tokens: cheapest and fastest) or an effort level,
    "low", "medium" or "high", for adaptive thinking. Recent models think by default,
    and their reasoning counts against `max_tokens` and is billed as output.
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-5-5", client=None, max_tokens: int | None = None,
                 thinking: str = "off"):
        if thinking not in ("off", "low", "medium", "high"):
            raise ValueError(f"thinking must be off, low, medium or high, not {thinking!r}")
        self.thinking = thinking
        self.label = model if thinking == "off" else f"{model}-thinking-{thinking}"
        max_tokens = max_tokens or (400 if thinking == "off" else 8000)
        if client is None:
            import anthropic  # optional dependency: pip install anthropic

            workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
            headers = {"anthropic-workspace-id": workspace} if workspace else None
            client = anthropic.Anthropic(max_retries=5, default_headers=headers)  # backs off on rate limits
        self.client, self.model, self.max_tokens = client, model, max_tokens

    def ask(self, prompt: str, image: bytes | None = None, schema: dict | None = None) -> Answer:
        start = time.perf_counter()
        output_config: dict = {"format": {"type": "json_schema", "schema": schema or SCHEMA}}
        if self.thinking == "off":
            thinking = {"type": "disabled"}
        else:
            thinking = {"type": "adaptive"}
            output_config["effort"] = self.thinking
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": _content(prompt, image)}],
            extra_body={"output_config": output_config, "thinking": thinking},
        )
        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise RuntimeError(f"no answer in the response (stop reason {response.stop_reason}); "
                               "raise max_tokens or turn thinking off")
        usage = getattr(response, "usage", None)
        return Answer(json.loads(text), time.perf_counter() - start,
                      getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))


def _content(prompt: str, image: bytes | None):
    if image is None:
        return prompt
    data = base64.b64encode(image).decode("ascii")
    return [{"type": "image", "source": {"type": "base64", "media_type": _media_type(image), "data": data}},
            {"type": "text", "text": prompt}]


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:3b", host: str | None = None, timeout: float = 300):
        self.model = self.label = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.timeout = timeout

    def ask(self, prompt: str, image: bytes | None = None, schema: dict | None = None) -> Answer:
        user: dict = {"role": "user", "content": prompt}
        if image is not None:  # needs a vision model, such as qwen2.5vl
            user["images"] = [base64.b64encode(image).decode("ascii")]
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, user],
            "format": schema or SCHEMA,
            "stream": False,
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        # A server on this machine is reached directly, never through an HTTP proxy.
        local = urlparse(self.host).hostname in {"localhost", "127.0.0.1", "::1"}
        opener = urllib.request.build_opener(*([urllib.request.ProxyHandler({})] if local else []))
        start = time.perf_counter()
        with opener.open(request, timeout=self.timeout) as response:
            reply = json.loads(response.read().decode("utf-8"))
        return Answer(json.loads(reply["message"]["content"]), time.perf_counter() - start,
                      reply.get("prompt_eval_count"), reply.get("eval_count"))


BACKENDS = {"anthropic": AnthropicBackend, "ollama": OllamaBackend}

# USD per million tokens (input, output), list prices for prompts under 100k tokens, October 2026.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """List-price cost of the given tokens, or None for a model without a known price (local ones)."""
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


class CachedBackend:
    """Answers stored per receipt under `cache_dir`, keyed by model and prompt."""

    def __init__(self, backend, cache_dir: Path):
        self.backend = backend
        label = getattr(backend, "label", getattr(backend, "model", "model"))
        model = label.replace("/", "_").replace(":", "_")
        self.dir = Path(cache_dir) / f"{backend.name}__{model}"
        self.calls = 0  # requests actually sent, cache hits excluded
        self.new_input_tokens = 0
        self.new_output_tokens = 0

    def ask(self, key: str, prompt: str, image: bytes | None = None, schema: dict | None = None) -> Answer:
        text = f"{PROMPT_VERSION}\n{SYSTEM}\n{prompt}"
        if schema is not None:
            text += "\n" + json.dumps(schema, sort_keys=True)
        h = hashlib.sha256(text.encode())
        if image is not None:
            h.update(image)
        digest = h.hexdigest()[:16]
        path = self.dir / (f"{key}.image.json" if image is not None else f"{key}.json")
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("prompt_sha") == digest:
                return Answer(saved["answer"], saved["seconds"], saved.get("input_tokens"),
                              saved.get("output_tokens"))
        answer = self.backend.ask(prompt, image, schema) if (image or schema) else self.backend.ask(prompt)
        self.calls += 1
        self.new_input_tokens += answer.input_tokens or 0
        self.new_output_tokens += answer.output_tokens or 0
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"prompt_sha": digest, "answer": answer.data, "seconds": answer.seconds,
                                    "input_tokens": answer.input_tokens,
                                    "output_tokens": answer.output_tokens}, indent=1), encoding="utf-8")
        return answer


@dataclass
class Checked:
    """One field of a model answer after checking it against the text."""

    value: Decimal | dt.date | None
    # "grounded", "wrong row", "not in the text", "read from the image", "unreadable" or "no answer"
    status: str
    row: int | None = None  # 0-based OCR row where the value is printed, if it is
    claimed: str = ""
    line: str = ""  # the line the model quoted, when it read the image


def _values(field_name: str, line: str) -> set:
    if field_name == "total":
        return {a.value for a in find_amounts(line)}
    return {d.value for d in find_dates(line)}


def check_field(field_name: str, value_text: str, row: int, lines: Sequence[str]) -> Checked:
    value_text = (value_text or "").strip()
    if not value_text:
        return Checked(None, "no answer")
    value = parse_amount(value_text) if field_name == "total" else parse_date(value_text)
    if value is None:
        return Checked(None, "unreadable", claimed=value_text)
    if 1 <= row <= len(lines) and value in _values(field_name, lines[row - 1]):
        return Checked(value, "grounded", row - 1, value_text)
    for i, line in enumerate(lines):
        if value in _values(field_name, line):
            return Checked(value, "wrong row", i, value_text)
    return Checked(None, "not in the text", claimed=value_text)


def check_image_field(field_name: str, value_text: str, line: str, lines: Sequence[str]) -> Checked:
    """A value read from the scan. There is no OCR row to hold it to, so it is kept if it
    parses, and the OCR row that also shows it is recorded when there is one."""
    value_text = (value_text or "").strip()
    if not value_text:
        return Checked(None, "no answer")
    value = parse_amount(value_text) if field_name == "total" else parse_date(value_text)
    if value is None:
        return Checked(None, "unreadable", claimed=value_text, line=line)
    row = next((i for i, text in enumerate(lines) if value in _values(field_name, text)), None)
    return Checked(value, "read from the image", row, value_text, line)


def check_image_answer(data: dict, lines: Sequence[str]) -> dict[str, Checked]:
    return {f: check_image_field(f, str(data.get(f, "")), str(data.get(f"{f}_line", "")), lines)
            for f in ("total", "date")}


def check_answer(data: dict, lines: Sequence[str]) -> dict[str, Checked]:
    return {f: check_field(f, str(data.get(f, "")), _as_int(data.get(f"{f}_row")), lines)
            for f in ("total", "date")}


def _as_int(x) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


@dataclass
class Routed:
    """The cascade's decision for one receipt."""

    total: Extraction | None
    date: Extraction | None
    route: str  # "confirmed by cash", "confident ranker" or "llm"
    llm: dict[str, Checked] = field(default_factory=dict)
    answer: Answer | None = None


def consult_text(ask, lines: Sequence[str]):
    """Ask about the OCR rows; returns the answer and its fields checked against the text."""
    answer = ask(build_prompt(lines))
    return answer, check_answer(answer.data, lines)


def consult_image(ask, image: bytes, lines: Sequence[str]):
    """Ask about the scan itself; the OCR text is only used to note where a value also appears."""
    answer = ask(VISION_INSTRUCTIONS, image, VISION_SCHEMA)
    return answer, check_image_answer(answer.data, lines)


def cascade(lines: Sequence[str], rules_total: Extraction | None, rules_date: Extraction | None,
            ranked: Extraction | None, consult, threshold: float = 0.95) -> Routed:
    """Accept the cheap answer when it is trustworthy, ask the model otherwise.

    The total is trusted when the rules' total matches cash minus change, or when the
    ranker is confident (score at or above `threshold`). The model is consulted when the
    total is not trusted, or when the rules found no date. `consult()` returns the model's
    answer and its checked fields (`consult_text` or `consult_image`).

    When the model is consulted, its accepted total replaces the ranker's (only if the
    total was not trusted), and its accepted date replaces the rules' date.
    """
    if rules_total is not None and rules_total.checked:
        trusted, route = rules_total, "confirmed by cash"
    elif ranked is not None and ranked.score is not None and ranked.score >= threshold:
        trusted, route = ranked, "confident ranker"
    else:
        trusted, route = None, "llm"
    if trusted is not None and rules_date is not None:
        return Routed(trusted, rules_date, route)

    answer, checked = consult()
    paid = cash_minus_change(lines)

    def extraction(c: Checked) -> Extraction:
        row = -1 if c.row is None else c.row
        evidence = c.line if c.row is None else lines[c.row]
        return Extraction(c.value, row, evidence, f"llm ({c.status})")

    total = trusted if trusted is not None else ranked
    t = checked["total"]
    if trusted is None and t.value is not None:
        e = extraction(t)
        total = Extraction(e.value, e.line, e.evidence, e.rule, None if paid is None else paid == t.value)
    d = checked["date"]
    date = extraction(d) if d.value is not None else rules_date
    return Routed(total, date, route if trusted is None else "llm for the date", checked, answer)
