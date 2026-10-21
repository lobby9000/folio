"""Score search over the Tobacco3482 OCR text with questions written for sampled pages.

benchmarks/search_questions.jsonl holds one line per page looked at. Pages with at least
400 characters of OCR text were shuffled with seed 5 and taken in that order; a page was
skipped when its text was unreadable, said nothing specific, or was a template that many
other pages repeat, and the file keeps the reason. For each of the 200 pages kept, Claude
read the text and wrote one question, alternating two kinds:

* exact: it uses an identifier from the page (a name, a number, a code), the rest in
  other words;
* described: no names, numbers or codes at all, only what the page is about.

No question copies more than three consecutive words of its page apart from identifiers,
and each has a Spanish translation, so the same need is asked in two languages over
English pages. The page answers its question, and so does a copy of it or an equivalent
form: the pages most similar to each source page were read to find those.

Three methods rank pages (see search.py). Four ways of fusing their rankings with RRF,
each with three values of k, are compared on the 50 dev questions (both languages) by
MRR@10, and the best is then scored with the rest on the 150 test questions.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .evaluate import exact_mcnemar
from .metrics import wilson_interval
from .search import BM25, CharTfidf, Chunk, Dense, rank_docs, rrf, text_key, tokenize

BASE = ("BM25", "character n-grams", "embeddings")
SHORT = {"BM25": "BM25", "character n-grams": "char", "embeddings": "embeddings"}
FUSIONS = (("BM25", "embeddings"), ("character n-grams", "embeddings"), ("BM25", "character n-grams"), BASE)
RRF_KS = (10, 30, 60)
LANGS = ("en", "es")
LANG_NAMES = {"en": "English", "es": "Spanish"}
CUTOFFS = (1, 5, 10, 20)
DEPTH = 100


@dataclass(frozen=True)
class Question:
    id: str
    doc: str
    kind: str  # "exact" or "described"
    split: str  # "dev" or "test"
    relevant: frozenset[str]
    en: str
    es: str


def load_questions(path: Path) -> tuple[list[Question], list[dict]]:
    """The questions, and the pages that were looked at but skipped."""
    questions, skipped = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "skip" in row:
            skipped.append(row)
        else:
            questions.append(Question(row["id"], row["doc"], row["kind"], row["split"],
                                      frozenset(row["relevant"]), row["en"], row["es"]))
    return questions, skipped


def fusion_name(parts: Sequence[str], k: int) -> str:
    return " + ".join(SHORT[p] for p in parts) + f" (RRF, k={k})"


def first_hit(ranking: Sequence[str], relevant: Iterable[str]) -> int | None:
    """1-based position of the first right page, or None if it is not in the ranking."""
    relevant = set(relevant)
    return next((i for i, doc in enumerate(ranking, start=1) if doc in relevant), None)


def recall_at(ranks: Sequence[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks) if ranks else float("nan")


def mrr_at(ranks: Sequence[int | None], k: int = 10) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks) if ranks else float("nan")


class Searcher:
    """The three rankers over one set of chunks."""

    def __init__(self, chunks: Sequence[Chunk], vectors: dict[str, np.ndarray] | None = None):
        self.docs = [c.doc for c in chunks]
        texts = [c.text for c in chunks]
        self.bm25 = BM25(texts)
        self.char = CharTfidf(texts)
        self.vectors = vectors or {}
        missing = [c.id for c in chunks if text_key(c.text) not in self.vectors]
        self.dense = None if missing else Dense(np.stack([self.vectors[text_key(t)] for t in texts]))

    def scores(self, method: str, query: str) -> np.ndarray:
        if method == "BM25":
            return self.bm25.scores(query)
        if method == "character n-grams":
            return self.char.scores(query)
        if method == "embeddings":
            if self.dense is None or text_key(query) not in self.vectors:
                raise KeyError("no vector for this text: run scripts/embed_search.py on chunks.jsonl and "
                               "queries.jsonl (folio export-search writes them)")
            return self.dense.scores(self.vectors[text_key(query)])
        raise ValueError(f"unknown method {method!r}")

    def rank(self, method: str, query: str, top: int = DEPTH) -> list[str]:
        return rank_docs(self.scores(method, query), self.docs, top)


@dataclass
class Row:
    question: str
    lang: str
    kind: str
    split: str
    method: str
    rank: int | None  # first right page, within the first DEPTH pages


@dataclass
class Report:
    pages: int
    chunks: int
    questions: list[Question]
    skipped: list[dict]
    rows: list[Row]
    selection: dict[str, float]  # fusion -> MRR@10 on the dev questions, both languages
    chosen: str
    overlap: dict[tuple[str, str], float]  # (kind, lang) -> share of question words on the page


def word_overlap(question: str, page: str) -> float:
    """Share of the question's content words (stemmed) that also appear on the page."""
    words = set(tokenize(question))
    return len(words & set(tokenize(page))) / len(words) if words else 0.0


def run(questions: Sequence[Question], chunks: Sequence[Chunk], pages: dict[str, str],
        vectors: dict[str, np.ndarray] | None = None, splits: Sequence[str] = ("dev", "test"),
        skipped: Sequence[dict] = (), log: Callable[[str], None] = lambda _: None) -> Report:
    log(f"indexing {len(chunks)} chunks")
    searcher = Searcher(chunks, vectors)
    methods = [m for m in BASE if m != "embeddings" or searcher.dense is not None]
    fusions = [p for p in FUSIONS if all(m in methods for m in p)]
    rows = []
    asked = [q for q in questions if q.split in splits]
    for n, q in enumerate(asked, start=1):
        if n % 50 == 0:
            log(f"  {n}/{len(asked)} questions")
        for lang in LANGS:
            text = getattr(q, lang)
            ranked = {m: searcher.rank(m, text) for m in methods}
            for m in methods:
                rows.append(Row(q.id, lang, q.kind, q.split, m, first_hit(ranked[m], q.relevant)))
            for parts in fusions:
                for k in RRF_KS:
                    fused = rrf([ranked[p] for p in parts], k=k, top=DEPTH)
                    rows.append(Row(q.id, lang, q.kind, q.split, fusion_name(parts, k),
                                    first_hit(fused, q.relevant)))
    candidates = [fusion_name(p, k) for p in fusions for k in RRF_KS]
    dev = [r for r in rows if r.split == "dev"]
    selection = {c: mrr_at([r.rank for r in dev if r.method == c]) for c in candidates}
    chosen = max(candidates, key=lambda c: (selection[c], -candidates.index(c))) if candidates else ""
    overlap = {(kind, lang): statistics.mean(word_overlap(getattr(q, lang), pages[q.doc])
                                             for q in questions if q.kind == kind)
               for kind in ("exact", "described") for lang in LANGS}
    return Report(len(pages), len(chunks), list(questions), list(skipped), rows, selection, chosen, overlap)


def write_csv(rows: Sequence[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in fields(Row)])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _ranks(rows: Sequence[Row], split: str, method: str, lang: str | None = None,
           kind: str | None = None) -> list[int | None]:
    return [r.rank for r in rows if r.split == split and r.method == method
            and (lang is None or r.lang == lang) and (kind is None or r.kind == kind)]


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _recall_ci(ranks: Sequence[int | None], k: int) -> str:
    hits = sum(r is not None and r <= k for r in ranks)
    lo, hi = wilson_interval(hits, len(ranks))
    return f"{_pct(hits / len(ranks))} [{_pct(lo)}, {_pct(hi)}]"


def _table(rows: Sequence[Row], split: str, methods: Sequence[str], lang: str) -> list[str]:
    out = ["| Method | Recall@1 | Recall@5 | Recall@10 [95% CI] | Recall@20 | MRR@10 |",
           "|---|---|---|---|---|---|"]
    for m in methods:
        ranks = _ranks(rows, split, m, lang)
        out.append(f"| {m} | {_pct(recall_at(ranks, 1))} | {_pct(recall_at(ranks, 5))} | "
                   f"{_recall_ci(ranks, 10)} | {_pct(recall_at(ranks, 20))} | {mrr_at(ranks):.3f} |")
    return out


def _found(rows: Sequence[Row], split: str, method: str, lang: str, k: int = 10) -> dict[str, bool]:
    return {r.question: r.rank is not None and r.rank <= k for r in rows
            if r.split == split and r.method == method and r.lang == lang}


def render_markdown(report: Report, header: str = "") -> str:
    rows = report.rows
    methods = [m for m in BASE if any(r.method == m for r in rows)]
    methods += [report.chosen] if report.chosen else []
    splits = [s for s in ("test", "dev") if any(r.split == s for r in rows)]
    main = splits[0]
    n_main = len({r.question for r in rows if r.split == main})
    lines = ["# Search over scanned pages", ""]
    if header:
        lines += [header, ""]
    skipped = report.skipped
    reasons = {}
    for s in skipped:
        reasons[s["skip"]] = reasons.get(s["skip"], 0) + 1
    lines += [f"{report.pages} pages split into {report.chunks} chunks of up to 1,200 characters; a page "
              "ranks by its best chunk. Questions: "
              f"{len(report.questions)}, written for pages drawn at random "
              f"({len(report.questions) + len(skipped)} looked at, {len(skipped)} skipped: "
              + ", ".join(f"{n} {why}" for why, n in sorted(reasons.items(), key=lambda x: -x[1]))
              + "). Recall@k is the share of questions whose page (or a copy of it) is among the first k "
              "pages returned; MRR@10 averages 1/position of that page, counting 0 past the tenth.", ""]

    if report.selection:
        lines += ["## Choosing the fusion on the dev questions", "",
                  "MRR@10 over the 50 dev questions asked in English and in Spanish (100 queries). "
                  "Reciprocal Rank Fusion adds 1/(k + position) from each list.", "",
                  "| Fused methods | " + " | ".join(f"k={k}" for k in RRF_KS) + " |",
                  "|---|" + "---|" * len(RRF_KS)]
        for parts in FUSIONS:
            names = [fusion_name(parts, k) for k in RRF_KS]
            if names[0] not in report.selection:
                continue
            cells = [f"**{report.selection[n]:.3f}**" if n == report.chosen else f"{report.selection[n]:.3f}"
                     for n in names]
            lines.append(f"| {' + '.join(SHORT[p] for p in parts)} | " + " | ".join(cells) + " |")
        lines += ["", f"Chosen: `{report.chosen}`.", ""]

    title = "Test" if main == "test" else "Dev"
    lines += [f"## {title}: {n_main} questions", ""]
    for lang in LANGS:
        lines += [f"**Asked in {LANG_NAMES[lang]}**", ""] + _table(rows, main, methods, lang) + [""]

    lines += ["## By kind of question", "",
              f"Recall@10 on the {main} questions. Exact questions name something on the page; described "
              "ones only say what it is about.", "",
              "| Method | " + " | ".join(f"{kind}, {LANG_NAMES[lang]}" for lang in LANGS
                                       for kind in ("exact", "described")) + " |",
              "|---|" + "---|" * 4]
    for m in methods:
        cells = [_pct(recall_at(_ranks(rows, main, m, lang, kind), 10)) for lang in LANGS
                 for kind in ("exact", "described")]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    if "embeddings" in methods and report.chosen:
        lines += ["", "## What each method finds that the other misses", "",
                  f"{main.capitalize()} questions whose page is in the first 10 results.", "",
                  "| Language | BM25 only | embeddings only | both | neither | "
                  f"`{report.chosen}` |", "|---|---|---|---|---|---|"]
        for lang in LANGS:
            bm, de = _found(rows, main, "BM25", lang), _found(rows, main, "embeddings", lang)
            fu = _found(rows, main, report.chosen, lang)
            only_b = sum(bm[q] and not de[q] for q in bm)
            only_d = sum(de[q] and not bm[q] for q in bm)
            both = sum(bm[q] and de[q] for q in bm)
            neither = len(bm) - only_b - only_d - both
            lines.append(f"| {LANG_NAMES[lang]} | {only_b} | {only_d} | {both} | {neither} "
                         f"| {sum(fu.values())} of {len(fu)} |")
        lines += ["", f"Exact McNemar tests on Recall@10, `{report.chosen}` against each method alone:", ""]
        for lang in LANGS:
            fu = _found(rows, main, report.chosen, lang)
            parts = []
            for m in BASE:
                if m not in methods:
                    continue
                other = _found(rows, main, m, lang)
                b = sum(fu[q] and not other[q] for q in fu)
                c = sum(other[q] and not fu[q] for q in fu)
                parts.append(f"{m}: {b} gained, {c} lost, p = {exact_mcnemar(b, c):.2g}")
            lines.append(f"* {LANG_NAMES[lang]}: " + "; ".join(parts) + ".")

    if main == "test" and "dev" in splits:
        lines += ["", "## Dev questions, for reference", ""]
        for lang in LANGS:
            lines += [f"**Asked in {LANG_NAMES[lang]}**", ""] + _table(rows, "dev", methods, lang) + [""]

    lines += ["", "## How much the questions borrow from their pages", "",
              "Share of a question's content words (stop words dropped, stemmed) that appear on its page, "
              f"averaged over all {len(report.questions)} questions. Spanish questions share almost nothing "
              "but names and numbers.",
              "", "| Kind | English | Spanish |", "|---|---|---|"]
    for kind in ("exact", "described"):
        en, es = report.overlap[(kind, "en")], report.overlap[(kind, "es")]
        lines.append(f"| {kind} | {_pct(en)} | {_pct(es)} |")
    return "\n".join(lines).rstrip() + "\n"


def export_inputs(chunks: Sequence[Chunk], questions: Sequence[Question], folder: Path) -> tuple[Path, Path]:
    """chunks.jsonl and queries.jsonl for scripts/embed_search.py."""
    folder.mkdir(parents=True, exist_ok=True)
    chunk_path, query_path = folder / "chunks.jsonl", folder / "queries.jsonl"
    with chunk_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c.id, "text": c.text}, ensure_ascii=False) + "\n")
    with query_path.open("w", encoding="utf-8") as f:
        for q in questions:
            for lang in LANGS:
                row = {"id": f"{q.id}-{lang}", "text": getattr(q, lang)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return chunk_path, query_path

