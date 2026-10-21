"""Search over OCR text: BM25, character n-grams, embeddings and their fusion.

Stage 5 asks the retrieval half of RAG a plain question: given a question in words,
which scanned page answers it? Pages are split into chunks of up to 1,200 characters,
as in a RAG index, every method scores chunks, and a page takes the score of its best
chunk. Three families are compared:

* BM25 over stemmed words: exact terms, names and numbers.
* TF-IDF over character n-grams inside words: lexical, but tolerant of OCR errors.
* Dense embeddings: meaning, computed elsewhere and loaded as vectors.

Reciprocal Rank Fusion merges page rankings using only their ranks, so scores from
different methods never have to be put on one scale.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHUNK_CHARS = 1200
MIN_TAIL = 300


@dataclass(frozen=True)
class Chunk:
    id: str  # "<page id>#<k>"
    doc: str
    text: str


def chunk_text(text: str, max_chars: int = CHUNK_CHARS, min_tail: int = MIN_TAIL) -> list[str]:
    """Pieces of at most `max_chars`, cut at paragraph, then line, then word boundaries.

    A last piece shorter than `min_tail` (a signature, a cc list) joins the one before it,
    so a page never ends in a fragment that could match a question on its own.
    """
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"[ \t]+", " ", paragraph).strip()
        if not paragraph:
            continue
        units = paragraph.split("\n") if len(paragraph) > max_chars else [paragraph]
        for unit in units:
            unit = unit.strip()
            while len(unit) > max_chars:
                cut = unit.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                pieces.append(unit[:cut].strip())
                unit = unit[cut:].strip()
            if unit:
                pieces.append(unit)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current:
        chunks.append(current)
    if len(chunks) > 1 and len(chunks[-1]) < min_tail:
        tail = chunks.pop()
        chunks[-1] = f"{chunks[-1]}\n{tail}"
    return chunks


def make_chunks(docs: Sequence, max_chars: int = CHUNK_CHARS) -> list[Chunk]:
    """Chunks of every page that has text (`docs` are `classify.Doc`s)."""
    return [Chunk(f"{d.id}#{k}", d.id, text)
            for d in docs for k, text in enumerate(chunk_text(d.text, max_chars))]


_WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


def _stemmer():
    import snowballstemmer

    return snowballstemmer.stemmer("english")


STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "what which who whom whose when where why how did does do about there their they them these those".split()
)


def tokenize(text: str, stem=None) -> list[str]:
    """Lower-case words and numbers, stop words dropped, stemmed (Snowball English)."""
    stem = stem or _stemmer()
    words = [w for w in _WORD.findall(text.casefold()) if w not in STOPWORDS]
    return stem.stemWords(words)


class BM25:
    """Okapi BM25 over chunks, with the usual k1 and b."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        from sklearn.feature_extraction.text import CountVectorizer

        self.stem = _stemmer()
        self.vectorizer = CountVectorizer(analyzer=lambda t: tokenize(t, self.stem))
        tf = self.vectorizer.fit_transform(texts).tocsc().astype(np.float64)
        n = tf.shape[0]
        df = np.diff(tf.indptr)  # chunks containing each term (CSC: non-zeros per column)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        lengths = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1e-9))
        tf = tf.tocsr()
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        tf.data = tf.data * (k1 + 1) / (tf.data + norm[rows])
        self.weights = tf.tocsc()

    def scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        terms = q.indices
        if not len(terms):
            return np.zeros(self.weights.shape[0])
        return np.asarray(self.weights[:, terms] @ self.idf[terms]).ravel()


class CharTfidf:
    """Cosine similarity over TF-IDF of character 3-5 grams inside words."""

    def __init__(self, texts: Sequence[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
        self.matrix = self.vectorizer.fit_transform(texts)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray((self.matrix @ self.vectorizer.transform([query]).T).todense()).ravel()


class Dense:
    """Cosine similarity between precomputed vectors: one per chunk, one per query."""

    def __init__(self, chunk_vectors: np.ndarray):
        self.chunks = _normalise(np.asarray(chunk_vectors, dtype=np.float32))

    def scores(self, query_vector: np.ndarray) -> np.ndarray:
        return self.chunks @ _normalise(np.asarray(query_vector, dtype=np.float32)[None, :])[0]


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def text_key(text: str) -> str:
    """The key of a text's vector in embeddings.npz (scripts/embed_search.py uses the same)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_vectors(path: Path) -> dict[str, np.ndarray]:
    """Vectors written by scripts/embed_search.py, keyed by `text_key`."""
    with np.load(path) as saved:
        return dict(zip(saved["keys"].tolist(), saved["vectors"].astype(np.float32), strict=True))


def rank_docs(chunk_scores: np.ndarray, chunk_docs: Sequence[str], top: int = 100) -> list[str]:
    """Pages ordered by their best chunk, best first, at most `top` of them."""
    best: dict[str, float] = {}
    for i in np.argsort(-chunk_scores, kind="stable"):
        doc = chunk_docs[i]
        if doc not in best:
            best[doc] = float(chunk_scores[i])
            if len(best) == top:
                break
    return list(best)


def rrf(rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None,
        top: int = 100) -> list[str]:
    """Reciprocal Rank Fusion: each list adds weight / (k + rank) to every item in it."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + w / (k + rank)
    return sorted(fused, key=lambda item: (-fused[item], item))[:top]
