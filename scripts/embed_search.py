"""Dense vectors for stage 5, computed wherever the embedding model can be downloaded.

Reads chunks.jsonl and queries.jsonl from a folder (one {"id", "text"} object per line),
embeds every text with jina-embeddings-v2-base-es through fastembed (ONNX on the CPU)
and writes embeddings.npz to the same folder. Vectors are keyed by the SHA-1 of their
text, so running it again only embeds texts that are new or changed.

    pip install fastembed==0.9.0
    python scripts/embed_search.py data/search

This file does not import folio, so it runs in any environment that has fastembed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL = "jinaai/jina-embeddings-v2-base-es"
CACHE = Path.home() / ".cache" / "folio" / "fastembed"
SAVE_EVERY = 128
BATCH = 4  # long, noisy OCR chunks can become 1,000 tokens: small batches keep memory low


def key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_texts(folder: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for name in ("chunks.jsonl", "queries.jsonl"):
        path = folder / name
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    text = json.loads(line)["text"]
                    texts[key(text)] = text
    return texts


def load(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path) as saved:
        if str(saved["model"]) != MODEL:
            return {}
        return dict(zip(saved["keys"].tolist(), saved["vectors"], strict=True))


def save(path: Path, vectors: dict[str, np.ndarray]) -> None:
    tmp = path.with_name(path.stem + ".partial.npz")
    keys = sorted(vectors)
    np.savez(tmp, model=np.array(MODEL), keys=np.array(keys),
             vectors=np.stack([vectors[k] for k in keys]).astype(np.float16))
    os.replace(tmp, path)


def main(folder: Path) -> None:
    from fastembed import TextEmbedding

    out = folder / "embeddings.npz"
    texts = read_texts(folder)
    vectors = {k: v for k, v in load(out).items() if k in texts}
    todo = sorted((k for k in texts if k not in vectors), key=lambda k: len(texts[k]))
    print(f"{len(texts)} texts, {len(vectors)} already embedded, {len(todo)} to go", flush=True)
    if not todo:
        save(out, vectors)
        print(f"Nothing new. {out} is up to date.")
        return
    print(f"Loading {MODEL} (the first time it downloads about 640 MB to {CACHE})", flush=True)
    model = TextEmbedding(MODEL, cache_dir=str(CACHE))
    start = time.time()
    for done, (k, vector) in enumerate(zip(todo, model.embed([texts[k] for k in todo], batch_size=BATCH),
                                           strict=True), start=1):
        vectors[k] = np.asarray(vector, dtype=np.float32)
        if done % SAVE_EVERY == 0 or done == len(todo):
            save(out, vectors)
            elapsed = time.time() - start
            left = elapsed / done * (len(todo) - done)
            print(f"  {done}/{len(todo)} embedded, {elapsed / 60:.1f} min so far, "
                  f"about {left / 60:.1f} min left", flush=True)
    size = out.stat().st_size / 1e6
    print(f"Done: {len(vectors)} vectors in {out} ({size:.1f} MB, {time.time() - start:.0f} s).")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent)
