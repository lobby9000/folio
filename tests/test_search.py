"""Search on invented pages: chunking, the three rankers, fusion, the evaluation and its
report, the question set itself, and the embedding script run with a stand-in model."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from folio.classify import Doc
from folio.evaluate_search import (
    BASE,
    Question,
    Searcher,
    export_inputs,
    first_hit,
    fusion_name,
    load_questions,
    mrr_at,
    recall_at,
    render_markdown,
    run,
    word_overlap,
)
from folio.search import (
    BM25,
    CharTfidf,
    Chunk,
    Dense,
    chunk_text,
    load_vectors,
    make_chunks,
    rank_docs,
    rrf,
    text_key,
    tokenize,
)

ROOT = Path(__file__).resolve().parents[1]

PAGES = {
    "Memo/1": "MEMORANDUM\nSubject: nicotine overspray in the explosion proof room at Semiworks.\n"
              "The spray cylinder will be installed next week.",
    "Letter/2": "Dear Dr. Hickey, thank you for your publications on hormesis in hydroids and growth.",
    "Memo/3": "Records Management Program: the Blend Sheets B1082 are under disposal suspension.",
    "Email/4": "The meeting with the youth smoking prevention team has been rescheduled to October.",
    "Report/5": "Smoking in Sweden: per capita cigarette consumption rose while men turned to snuff.",
}


def docs():
    return [Doc(i, i.split("/")[0], t) for i, t in PAGES.items()]


def test_chunks_respect_the_limit_keep_every_word_and_absorb_short_tails():
    text = "\n\n".join(f"Paragraph {k} " + "word " * 40 for k in range(10)) + "\n\nSigned, J. Smith"
    chunks = chunk_text(text, max_chars=500, min_tail=100)
    assert all(len(c) <= 600 for c in chunks)
    assert " ".join(chunks).split() == text.split()
    assert not chunks[-1].startswith("Signed")  # the signature joined the chunk before it
    long_line = "x" * 50 + " " + "y" * 900
    assert all(len(c) <= 500 for c in chunk_text(long_line, max_chars=500))
    assert chunk_text("   \n\n  ") == []
    chunks = make_chunks(docs())
    assert {c.doc for c in chunks} == set(PAGES) and chunks[0].id == "Memo/1#0"


def test_tokenize_drops_stop_words_and_stems():
    words = tokenize("What were the nicotine levels reported in 1986?")
    assert words == ["nicotin", "level", "report", "1986"]
    assert tokenize("B1082 and 12487.013") == ["b1082", "12487.013"]


def test_bm25_prefers_rare_terms_and_shorter_matches():
    texts = ["smoke smoke smoke report", "smoke hydroids report", "report " * 30 + "hydroids"]
    bm25 = BM25(texts)
    s = bm25.scores("hydroids")
    assert s[0] == 0 and s[1] > s[2] > 0  # same term count, the shorter chunk wins
    s = bm25.scores("smoke hydroids")
    assert np.argmax(s) == 1  # the rare term outweighs repeats of a common one
    assert not bm25.scores("zzz").any()


def test_character_ngrams_survive_ocr_errors():
    char = CharTfidf(["nicotine overspray at semiworks", "hormesis in hydroids", "records retention",
                      "nicotine levels", "hydroids grow"])
    assert np.argmax(char.scores("nicotme ovenspray")) == 0


def test_dense_is_cosine_and_pages_take_their_best_chunk():
    dense = Dense(np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]))
    assert dense.scores(np.array([0.0, 5.0])) == pytest.approx([0.0, 1.0, np.sqrt(0.5)])
    ranked = rank_docs(np.array([0.1, 0.9, 0.5, 0.95]), ["a", "b", "b", "c"], top=2)
    assert ranked == ["c", "b"]


def test_rrf_rewards_agreement_and_breaks_ties_by_name():
    assert rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)[:2] == ["a", "b"]
    assert rrf([["x"], ["y"]], k=10) == ["x", "y"]
    # a page fourth in both lists against a page first in one: k decides which matters more
    lists = [["a", "x1", "x2", "b"], ["y1", "y2", "y3", "b"]]
    assert rrf(lists, k=1)[0] == "a" and rrf(lists, k=60)[0] == "b"
    assert rrf([["a"], ["b"]], weights=[1, 2]) == ["b", "a"]


def test_ranks_recall_and_mrr():
    assert first_hit(["a", "b", "c"], {"c", "z"}) == 3 and first_hit(["a"], {"b"}) is None
    ranks = [1, 3, None, 12]
    assert recall_at(ranks, 1) == 0.25 and recall_at(ranks, 10) == 0.5 and recall_at(ranks, 20) == 0.75
    assert mrr_at(ranks) == pytest.approx((1 + 1 / 3) / 4)


def fake_vectors(chunks, questions):
    """Vectors that point every question at its own page, and a little at another one."""
    order = sorted(PAGES)
    vec = {}
    for c in chunks:
        v = np.zeros(len(order) + 1, dtype=np.float32)
        v[order.index(c.doc)] = 1.0
        v[-1] = 0.2
        vec[text_key(c.text)] = v
    for q in questions:
        for lang in ("en", "es"):
            v = np.zeros(len(order) + 1, dtype=np.float32)
            v[order.index(q.doc)] = 1.0
            vec[text_key(getattr(q, lang))] = v
    return vec


QUESTIONS = [
    Question("q001", "Memo/1", "exact", "dev", frozenset({"Memo/1"}),
             "Where will the nicotine spray cylinder go at Semiworks?",
             "¿Dónde va el cilindro de rocío de nicotina en Semiworks?"),
    Question("q002", "Letter/2", "described", "dev", frozenset({"Letter/2"}),
             "Which letter thanks someone for papers on small doses that boost growth?",
             "¿Qué carta agradece artículos sobre dosis pequeñas que favorecen el crecimiento?"),
    Question("q003", "Memo/3", "exact", "test", frozenset({"Memo/3"}),
             "Why are the Blend Sheets B1082 kept?", "¿Por qué se conservan los Blend Sheets B1082?"),
    Question("q004", "Report/5", "described", "test", frozenset({"Report/5"}),
             "Which briefing says Nordic men moved to oral tobacco?",
             "¿Qué informe dice que los hombres nórdicos se pasaron al tabaco oral?"),
]


def test_run_chooses_the_fusion_on_dev_and_reports_test():
    chunks = make_chunks(docs())
    vectors = fake_vectors(chunks, QUESTIONS)
    report = run(QUESTIONS, chunks, dict(PAGES), vectors, ("dev", "test"),
                 skipped=[{"n": 9, "doc": "Form/9", "skip": "unreadable", "reason": "garbled"}])
    methods = {r.method for r in report.rows}
    assert set(BASE) <= methods and len(methods) == 3 + 4 * 3
    assert report.selection[report.chosen] == max(report.selection.values())
    assert all(r.rank == 1 for r in report.rows if r.method == "embeddings")
    md = render_markdown(report, "header")
    for part in ("## Choosing the fusion on the dev questions", "## Test: 2 questions",
                 "**Asked in Spanish**", "## By kind of question", "McNemar",
                 "## What each method finds that the other misses", "## Dev questions, for reference",
                 "## How much the questions borrow", "1 unreadable"):
        assert part in md
    dev_only = run(QUESTIONS, chunks, dict(PAGES), None, ("dev",))
    assert {r.method for r in dev_only.rows} == {"BM25", "character n-grams", fusion_name(BASE[:2], 10),
                                                  fusion_name(BASE[:2], 30), fusion_name(BASE[:2], 60)}
    assert "## Dev: 2 questions" in render_markdown(dev_only)


def test_missing_vectors_are_reported_not_guessed():
    chunks = make_chunks(docs())
    searcher = Searcher(chunks, fake_vectors(chunks, QUESTIONS))
    with pytest.raises(KeyError):
        searcher.scores("embeddings", "a question nobody embedded")
    assert Searcher(chunks, {}).dense is None


def test_word_overlap_counts_stems():
    assert word_overlap("nicotine sprays", "Nicotine overspray and spray cylinders") == 1.0
    assert word_overlap("hormesis", "nothing here") == 0.0


def test_question_set_follows_its_protocol():
    questions, skipped = load_questions(ROOT / "benchmarks" / "search_questions.jsonl")
    assert len(questions) == 200 and len({q.id for q in questions}) == 200
    assert len({q.doc for q in questions}) == 200 and all(q.doc in q.relevant for q in questions)
    assert [q.kind for q in questions] == ["exact", "described"] * 100
    dev = [q for q in questions if q.split == "dev"]
    assert len(dev) == 50 and sum(q.kind == "exact" for q in dev) == 25
    assert all(q.en.strip() and q.es.strip() for q in questions)
    assert {s["skip"] for s in skipped} <= {"unreadable", "no content", "template"}
    lines = (ROOT / "benchmarks" / "search_questions.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [r["n"] for r in rows] == list(range(len(rows)))  # every page looked at, in sample order


def load_script():
    spec = importlib.util.spec_from_file_location("embed_search", ROOT / "scripts" / "embed_search.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedding_script_resumes_and_matches_folio_keys(tmp_path, monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, name, cache_dir=None):
            assert name == "jinaai/jina-embeddings-v2-base-es"

        def embed(self, texts, batch_size=16):
            calls.append(len(texts))
            for t in texts:
                yield np.full(4, len(t), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeModel))
    chunks = make_chunks(docs())
    questions = QUESTIONS[:2]
    export_inputs(chunks, questions, tmp_path)
    script = load_script()
    assert script.key("¿Qué?") == text_key("¿Qué?")
    script.main(tmp_path)
    vectors = load_vectors(tmp_path / "embeddings.npz")
    assert len(vectors) == len(chunks) + 4 and vectors[text_key(chunks[0].text)][0] == len(chunks[0].text)
    with (tmp_path / "queries.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "extra", "text": "one more question"}) + "\n")
    script.main(tmp_path)
    assert calls == [len(chunks) + 4, 1]  # the second run only embeds the new text
    assert len(load_vectors(tmp_path / "embeddings.npz")) == len(chunks) + 5


def test_chunk_type_is_plain_data():
    c = Chunk("a#0", "a", "text")
    assert (c.id, c.doc, c.text) == ("a#0", "a", "text")
