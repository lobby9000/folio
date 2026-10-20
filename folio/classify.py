"""Document type from OCR text.

SROIE holds only receipts, so the classification half of stage 4 uses another public
set: QS-OCR-Small, the Tesseract text of the Tobacco3482 documents (Quicksign, 2019).
It has 3,482 scanned pages in ten classes, all from the same archive, so a class
cannot be told apart by where its pages came from, only by what they say.

The OCR text is noisy ("Fotal", "‘i986"), which is the point: a classifier for
incoming documents only ever sees what OCR read. Two kinds of features are compared:
word n-grams, and character n-grams inside words, which still match when OCR changes
a letter. The model is a linear SVM, and its margin (how far the best class is ahead
of the second) is a confidence that can decide which documents go to an LLM.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Folder names in QS-OCR-Small, and what each class is, for the reports and the LLM prompt.
CLASSES = {
    "ADVE": "advertisement: an ad or other promotional material",
    "Email": "email: a printed email message",
    "Form": "form: a form with fields, boxes or blanks to fill in",
    "Letter": "letter: a letter with a salutation and a signature",
    "Memo": "memo: an internal memorandum, usually with To, From and Subject",
    "News": "news: a newspaper or magazine article",
    "Note": "note: a short, often handwritten note",
    "Report": "report: a report, with sections and findings",
    "Resume": "resume: a curriculum vitae",
    "Scientific": "scientific: a scientific paper or technical publication",
}


@dataclass(frozen=True)
class Doc:
    id: str  # "<class>/<file name without .txt>"
    label: str
    text: str


def load_docs(root: Path) -> list[Doc]:
    """Every `<class>/<name>.txt` under `root`, in a stable order."""
    docs = []
    for path in sorted(Path(root).glob("*/*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        docs.append(Doc(f"{path.parent.name}/{path.stem}", path.parent.name, text))
    return docs


def split_docs(docs: Sequence[Doc], test_share: float = 0.2, seed: int = 0) -> dict[str, list[Doc]]:
    """Stratified train and test sets: the same share of every class is held out, the
    same way on every run. Model choice only ever looks at `train`."""
    rng = np.random.default_rng(seed)
    held = set()
    for label in sorted({d.label for d in docs}):
        ids = sorted(d.id for d in docs if d.label == label)
        k = int(round(len(ids) * test_share))
        held.update(ids[i] for i in rng.permutation(len(ids))[:k])
    return {"train": [d for d in docs if d.id not in held], "test": [d for d in docs if d.id in held]}


# The candidates compared by cross-validation. C is the SVM's inverse regularisation.
KINDS = {
    "naive Bayes, words": {"features": "words", "model": "nb"},
    "linear SVM, words, C=0.5": {"features": "words", "model": "svm", "C": 0.5},
    "linear SVM, words, C=2": {"features": "words", "model": "svm", "C": 2.0},
    "linear SVM, characters, C=0.5": {"features": "chars", "model": "svm", "C": 0.5},
    "linear SVM, characters, C=2": {"features": "chars", "model": "svm", "C": 2.0},
    "linear SVM, words and characters, C=0.5": {"features": "both", "model": "svm", "C": 0.5},
}


def _features(kind: str):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_union

    def words():
        return TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200_000)

    def chars():
        return TfidfVectorizer(sublinear_tf=True, min_df=2, analyzer="char_wb", ngram_range=(2, 5),
                               max_features=300_000)

    return {"words": words, "chars": chars, "both": lambda: make_union(words(), chars())}[kind]()


def make_classifier(kind: str):
    """A fresh, unfitted pipeline from OCR text to a class."""
    from sklearn.naive_bayes import ComplementNB
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    spec = KINDS[kind]
    if spec["model"] == "nb":
        return make_pipeline(_features(spec["features"]), ComplementNB(alpha=0.3))
    return make_pipeline(_features(spec["features"]), LinearSVC(C=spec["C"]))


def scores(model, texts: Sequence[str]) -> np.ndarray:
    """One score per class and document: the SVM's decision values, or log-probabilities."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(list(texts)))
    return np.asarray(model.predict_log_proba(list(texts)))


def predict(model, texts: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """The predicted class of each text, and its margin: how far the best class is ahead
    of the second, a confidence for routing."""
    s = scores(model, texts)
    top2 = np.sort(s, axis=1)[:, -2:]
    return [str(c) for c in model.classes_[s.argmax(axis=1)]], top2[:, 1] - top2[:, 0]


def top_terms(model, n: int = 6) -> dict[str, list[str]]:
    """The word n-grams that push hardest towards each class (word-feature SVMs only)."""
    vectorizer, svm = model[0], model[-1]
    if not hasattr(vectorizer, "get_feature_names_out") or not hasattr(svm, "coef_"):
        raise ValueError("top terms need a linear model on word features")
    names = vectorizer.get_feature_names_out()
    return {str(c): [str(names[i]) for i in np.argsort(-svm.coef_[k])[:n]]
            for k, c in enumerate(svm.classes_)}


# The LLM, asked to choose a class from the same OCR text.
SYSTEM = ("You sort scanned business documents by type. You read their OCR text, which has recognition "
          "errors. Answer only through the JSON schema.")

PROMPT = (
    "OCR text of one scanned document (it may be cut short):\n\n<document>\n{text}\n</document>\n\n"
    "Which kind of document is it? The kinds are:\n{kinds}\n"
    "Choose the single best kind."
)

SCHEMA = {
    "type": "object",
    "properties": {"kind": {"type": "string", "enum": [c.split(":")[0] for c in CLASSES.values()]}},
    "required": ["kind"],
    "additionalProperties": False,
}

_BY_NAME = {c.split(":")[0]: label for label, c in CLASSES.items()}


def build_prompt(text: str, max_chars: int = 3000) -> str:
    """The prompt for one document. Long pages are cut, since the start of a page says the most
    about its kind and every character costs tokens."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())[:max_chars]
    kinds = "\n".join(f"- {c}" for c in CLASSES.values())
    return PROMPT.format(text=text or "(OCR found no text on this page)", kinds=kinds)


def parse_kind(data: dict) -> str | None:
    """The class folder name for the model's answer, or None if it is not one of the kinds."""
    return _BY_NAME.get(str(data.get("kind", "")).strip().casefold())
