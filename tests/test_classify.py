"""Document-type classification on a small synthetic corpus: loading, the stratified split,
the models, the routing threshold, the LLM prompt and the evaluation."""

import random
from pathlib import Path

import numpy as np
import pytest

from folio.classify import (
    CLASSES,
    KINDS,
    SCHEMA,
    SYSTEM,
    build_prompt,
    load_docs,
    make_classifier,
    parse_kind,
    predict,
    split_docs,
    top_terms,
)
from folio.cli import main
from folio.evaluate_classify import ask_llm, render_markdown, routing_threshold, run
from folio.llm import Answer

VOCAB = {
    "Memo": ["memorandum", "to", "from", "subject", "date", "cc", "re", "budget", "department"],
    "Letter": ["dear", "sir", "sincerely", "yours", "letter", "regards", "thank", "enclosed", "truly"],
    "Email": ["from", "sent", "to", "subject", "email", "com", "forwarded", "message", "reply"],
}
NOISE = ["the", "of", "and", "tobacco", "company", "report", "page", "1984", "smoke", "brand"]


def corpus(per_class: int = 24, seed: int = 0) -> dict[str, list[str]]:
    rng = random.Random(seed)
    texts = {}
    for label, words in VOCAB.items():
        texts[label] = [" ".join(rng.choice(words if rng.random() < 0.6 else NOISE) for _ in range(40))
                        for _ in range(per_class)]
    return texts


def write_corpus(root: Path, per_class: int = 24) -> Path:
    for label, texts in corpus(per_class).items():
        (root / label).mkdir(parents=True)
        for k, text in enumerate(texts):
            (root / label / f"{label.lower()}{k:03d}.txt").write_text(text, encoding="utf-8")
    return root


def test_load_docs_reads_one_folder_per_class(tmp_path):
    docs = load_docs(write_corpus(tmp_path / "docs", per_class=3))
    assert len(docs) == 9 and docs[0].id == "Email/email000" and docs[0].label == "Email"
    assert {d.label for d in docs} == set(VOCAB) and all(d.text for d in docs)


def test_split_is_stratified_disjoint_and_repeatable(tmp_path):
    docs = load_docs(write_corpus(tmp_path / "docs", per_class=10))
    split = split_docs(docs, test_share=0.2, seed=0)
    assert len(split["test"]) == 6 and len(split["train"]) == 24
    assert {d.label for d in split["test"]} == set(VOCAB)
    assert all(sum(d.label == c for d in split["test"]) == 2 for c in VOCAB)
    assert not {d.id for d in split["train"]} & {d.id for d in split["test"]}
    assert [d.id for d in split_docs(docs, 0.2, 0)["test"]] == [d.id for d in split["test"]]
    assert [d.id for d in split_docs(docs, 0.2, 1)["test"]] != [d.id for d in split["test"]]


@pytest.mark.parametrize("kind", list(KINDS))
def test_every_candidate_learns_the_synthetic_classes(kind):
    texts = corpus(per_class=30)
    train = [(t, label) for label, ts in texts.items() for t in ts[:24]]
    held = [(t, label) for label, ts in texts.items() for t in ts[24:]]
    model = make_classifier(kind).fit([t for t, _ in train], [label for _, label in train])
    pred, margin = predict(model, [t for t, _ in held])
    assert np.mean([p == label for p, (_, label) in zip(pred, held, strict=True)]) >= 0.8
    assert len(margin) == len(held) and (margin >= 0).all()


def test_top_terms_name_the_words_of_each_class():
    texts = corpus(per_class=30)
    model = make_classifier("linear SVM, words, C=2").fit(
        [t for ts in texts.values() for t in ts], [label for label, ts in texts.items() for _ in ts])
    terms = top_terms(model, n=4)
    assert set(terms) == set(VOCAB)
    for label, words in VOCAB.items():  # each class is pushed by its own words, not the shared noise
        assert any(w in " ".join(terms[label]).split() for w in words)
    with pytest.raises(ValueError):
        top_terms(make_classifier("naive Bayes, words").fit(["a b", "a c"], ["Memo", "Email"]))


def test_routing_threshold_keeps_the_confident_documents():
    margins = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    correct = [True] * 6 + [False, True, False, False]
    assert routing_threshold(correct, margins, target=0.95) == pytest.approx(0.4)
    assert routing_threshold(correct, margins, target=0.86) == pytest.approx(0.2)
    assert routing_threshold([False, False], np.array([0.5, 0.1])) == float("inf")


def test_prompt_lists_the_kinds_and_cuts_long_pages():
    prompt = build_prompt("MEMORANDUM\n\n\n\nTO: All staff " + "x" * 5000, max_chars=200)
    assert "MEMORANDUM\n\nTO: All staff" in prompt and "x" * 300 not in prompt
    assert all(c.split(":")[0] in prompt for c in CLASSES.values())
    assert "no text" in build_prompt("   ")
    assert SCHEMA["properties"]["kind"]["enum"][0] == "advertisement" and "OCR" in SYSTEM


def test_parse_kind_maps_answers_back_to_folders():
    assert parse_kind({"kind": "memo"}) == "Memo" and parse_kind({"kind": " News "}) == "News"
    assert parse_kind({"kind": "advertisement"}) == "ADVE" and parse_kind({"kind": "invoice"}) is None
    assert parse_kind({}) is None


def test_evaluation_chooses_by_cross_validation_and_reports_the_llm(tmp_path):
    docs = load_docs(write_corpus(tmp_path / "docs", per_class=20))
    split = split_docs(docs, 0.25, 0)
    kinds = ["naive Bayes, words", "linear SVM, words, C=2"]
    report = run(split["train"], split["test"], kinds=kinds, folds=3, repeats=1)
    assert report.chosen in kinds and len(report.rows) == len(split["test"]) == 15
    assert set(report.test) == set(kinds) and report.majority in VOCAB
    assert all(r.final == r.predicted and isinstance(r.routed, bool) for r in report.rows)

    calls = []

    def ask(key, prompt, image=None, schema=None, system=None):
        calls.append(key)
        assert schema is SCHEMA and system is SYSTEM and "/" not in key
        truth = key.split("__")[0].lower()
        return Answer({"kind": truth}, 0.4, 900, 5)

    for row in report.rows:
        row.routed = row.doc.startswith("Memo")  # route a known set, whatever the threshold chose
    ask_llm(report.rows, split["test"], ask, mode="cascade")
    assert len(calls) == 5 and all(r.final_correct for r in report.rows if r.routed)
    ask_llm(report.rows, split["test"], ask, mode="all")
    assert len(calls) == 20 and all(r.llm_correct for r in report.rows)
    md = render_markdown(report, "header", "Cost: none.")
    for part in ("## Model choice", "## Test", "## By class", "Confusion matrix", "## Confidence",
                 "## The LLM", "LLM alone, every test document: 15 of 15", "## What the word model looks at"):
        assert part in md


def test_eval_classify_command_runs_without_tesseract(tmp_path, monkeypatch):
    data = write_corpus(tmp_path / "docs", per_class=15)
    monkeypatch.setattr("folio.cli.tesseract_available", lambda: False)
    out = tmp_path / "results"
    args = ["eval-classify", "--data", str(data), "--folds", "3", "--repeats", "1", "--out", str(out)]
    assert main(args) == 0
    assert (out / "classify_eval.md").exists() and (out / "classify_eval.csv").exists()
    assert main(["eval-classify", "--data", str(tmp_path / "missing"), "--out", str(out)]) == 2
