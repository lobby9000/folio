"""Ask a language model for the total and the date, with the row that proves each one.

The model sees the receipt's OCR rows, numbered, and must answer through a JSON
schema: for each field, the value as printed and the number of the row it read it
from. Nothing is taken on trust. An answer is kept only if the value, once
normalised, is actually printed in the OCR text, so the model can choose between
amounts on the receipt but cannot invent one. That is what "textual evidence" means
here, and it is what keeps a hallucinated total out of the results.

Two backends speak the same schema:

* `anthropic`: the Claude API with structured outputs. Needs `pip install anthropic`
  and an `ANTHROPIC_API_KEY`.
* `ollama`: a local model served by Ollama (`ollama pull qwen2.5:3b`), so nothing
  leaves the machine. It uses Ollama's structured outputs on `/api/chat`.

Every answer is cached on disk with the prompt's hash, so an evaluation can be
stopped and resumed, and rerunning it costs nothing.
"""

from __future__ import annotations

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


@dataclass
class Answer:
    data: dict
    seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-5-5", client=None, max_tokens: int = 300):
        if client is None:
            import anthropic  # optional dependency: pip install anthropic

            client = anthropic.Anthropic()
        self.client, self.model, self.max_tokens = client, model, max_tokens

    def ask(self, prompt: str) -> Answer:
        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}},
        )
        text = next(block.text for block in response.content if block.type == "text")
        usage = getattr(response, "usage", None)
        return Answer(json.loads(text), time.perf_counter() - start,
                      getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:3b", host: str | None = None, timeout: float = 300):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.timeout = timeout

    def ask(self, prompt: str) -> Answer:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "format": SCHEMA,
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


class CachedBackend:
    """Answers stored per receipt under `cache_dir`, keyed by model and prompt."""

    def __init__(self, backend, cache_dir: Path):
        self.backend = backend
        model = getattr(backend, "model", "model").replace("/", "_").replace(":", "_")
        self.dir = Path(cache_dir) / f"{backend.name}__{model}"
        self.calls = 0  # requests actually sent, cache hits excluded

    def ask(self, key: str, prompt: str) -> Answer:
        digest = hashlib.sha256(f"{PROMPT_VERSION}\n{SYSTEM}\n{prompt}".encode()).hexdigest()[:16]
        path = self.dir / f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("prompt_sha") == digest:
                return Answer(saved["answer"], saved["seconds"], saved.get("input_tokens"),
                              saved.get("output_tokens"))
        answer = self.backend.ask(prompt)
        self.calls += 1
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"prompt_sha": digest, "answer": answer.data, "seconds": answer.seconds,
                                    "input_tokens": answer.input_tokens,
                                    "output_tokens": answer.output_tokens}, indent=1), encoding="utf-8")
        return answer


@dataclass
class Checked:
    """One field of a model answer after checking it against the text."""

    value: Decimal | dt.date | None
    status: str  # "grounded", "wrong row", "not in the text", "unreadable", "no answer"
    row: int | None = None  # 0-based row where the value is printed
    claimed: str = ""


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


def cascade(lines: Sequence[str], rules_total: Extraction | None, rules_date: Extraction | None,
            ranked: Extraction | None, ask, threshold: float = 0.95) -> Routed:
    """Accept the cheap answer when it is trustworthy, ask the model otherwise.

    1. The rules' total matches cash minus change: keep it.
    2. The ranker is confident (score at or above `threshold`): keep its total.
    3. Otherwise ask the model (`ask(prompt) -> Answer`). A grounded total replaces the
       ranker's. The model's date is used only when the rules found none.
    """
    if rules_total is not None and rules_total.checked:
        return Routed(rules_total, rules_date, "confirmed by cash")
    if ranked is not None and ranked.score is not None and ranked.score >= threshold:
        return Routed(ranked, rules_date, "confident ranker")
    answer = ask(build_prompt(lines))
    checked = check_answer(answer.data, lines)
    total, date = ranked, rules_date
    paid = cash_minus_change(lines)
    t = checked["total"]
    if t.value is not None:
        total = Extraction(t.value, t.row, lines[t.row], f"llm ({t.status})",
                           None if paid is None else paid == t.value)
    d = checked["date"]
    if date is None and d.value is not None:
        date = Extraction(d.value, d.row, lines[d.row], f"llm ({d.status})")
    return Routed(total, date, "llm", checked, answer)
