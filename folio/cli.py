"""Command line: `folio ocr` for one image and `folio eval-ocr` for an experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pytesseract

from .evaluate import compare, evaluate, render_markdown, summarize, write_csv
from .ocr import OcrConfig, run_ocr, tesseract_available
from .preprocess import PIPELINES
from .sroie import load_receipts

DEFAULT_CONFIG = ("gray", 3)  # what you get with no preprocessing and Tesseract's default layout mode


def _cmd_ocr(args: argparse.Namespace) -> int:
    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read the image: {args.image}", file=sys.stderr)
        return 2
    result = run_ocr(PIPELINES[args.prep](img), OcrConfig(lang=args.lang, psm=args.psm))
    if args.words:
        for w in result.words:
            conf = "-" if w.conf is None else f"{w.conf:5.1f}"
            print(f"{conf}\t{int(w.left)},{int(w.top)},{int(w.right)},{int(w.bottom)}\t{w.text}")
    else:
        print(result.text)
    if result.mean_confidence is not None:
        print(f"\n[{len(result.words)} words, mean confidence {result.mean_confidence:.1f}]", file=sys.stderr)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    receipts = load_receipts(Path(args.data), limit=args.limit, seed=args.seed)
    if not receipts:
        print(f"No receipts found under {args.data} (expected img/, box/ and key/).", file=sys.stderr)
        return 2
    print(f"Evaluating {len(receipts)} receipts x {len(args.prep)} pipelines x {len(args.psm)} page modes "
          f"with {args.workers} worker(s)...", file=sys.stderr)
    rows = evaluate(receipts, args.prep, args.psm, lang=args.lang, workers=args.workers)
    out = Path(args.out)
    write_csv(rows, out / "ocr_eval.csv")
    summaries = summarize(rows)
    best = (summaries[0].pipeline, summaries[0].psm)
    configs = {(s.pipeline, s.psm) for s in summaries}
    comparison = None
    if DEFAULT_CONFIG in configs and best != DEFAULT_CONFIG:
        comparison = compare(rows, best, DEFAULT_CONFIG)
    header = (
        f"# OCR baseline on SROIE\n\n{len(receipts)} receipts sampled with seed {args.seed}, "
        f"Tesseract {pytesseract.get_tesseract_version()}, language `{args.lang}`, LSTM engine. "
        "Both texts are put in the same reading order and compared case-insensitively. "
        "Receipts whose key file leaves a field empty are left out of that field's rate. "
        "Best mean CER first."
    )
    report = render_markdown(summaries, comparison, header)
    (out / "ocr_eval.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="folio", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ocr = sub.add_parser("ocr", help="read one image and print its text")
    p_ocr.add_argument("image")
    p_ocr.add_argument("--prep", default="otsu", choices=sorted(PIPELINES))
    p_ocr.add_argument("--psm", type=int, default=4)
    p_ocr.add_argument("--lang", default="eng", help="Tesseract languages, e.g. spa or spa+eng")
    p_ocr.add_argument("--words", action="store_true", help="print each word with its box and confidence")
    p_ocr.set_defaults(func=_cmd_ocr)

    p_eval = sub.add_parser("eval-ocr", help="compare preprocessing and page modes on SROIE")
    p_eval.add_argument("--data", default="data/sroie/data", help="folder with img/, box/ and key/")
    p_eval.add_argument("--limit", type=int, default=100)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--prep", nargs="+", default=list(PIPELINES), choices=sorted(PIPELINES))
    p_eval.add_argument("--psm", nargs="+", type=int, default=[3, 4, 6])
    p_eval.add_argument("--lang", default="eng")
    p_eval.add_argument("--workers", type=int, default=2)
    p_eval.add_argument("--out", default="results")
    p_eval.set_defaults(func=_cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not tesseract_available():
        print("Tesseract is not installed. On Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-spa",
              file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
