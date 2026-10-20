"""Command line: `folio ocr` and `folio extract` for one image, `folio eval-ocr` and
`folio eval-extract` for the experiments on SROIE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pytesseract

from . import evaluate_extract, evaluate_learned
from .cache import DEFAULT_CACHE, ocr_receipts
from .evaluate import compare, evaluate, render_markdown, summarize, write_csv
from .extract import EXTRACTORS
from .learn import TotalRanker
from .ocr import OcrConfig, run_ocr, tesseract_available
from .preprocess import PIPELINES
from .sroie import SPLITS, load_receipts, load_split

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


_CHECK = {True: "matches cash minus change", False: "does not match cash minus change", None: ""}


def _cmd_extract(args: argparse.Namespace) -> int:
    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read the image: {args.image}", file=sys.stderr)
        return 2
    result = run_ocr(PIPELINES[args.prep](img), OcrConfig(lang=args.lang, psm=args.psm))
    lines = result.text.splitlines()
    found = {field: extract(lines) for field, extract in EXTRACTORS["rules"].items()}
    if args.ranker:
        found["total"] = TotalRanker.load(Path(args.ranker)).extract(result.rows)
    if args.json:
        print(json.dumps({
            field: None if e is None else {"value": str(e.value), "rule": e.rule, "row": e.line + 1,
                                           "evidence": e.evidence.strip(), "checked": e.checked,
                                           **({"score": round(e.score, 3)} if e.score is not None else {})}
            for field, e in found.items()
        }, indent=2, ensure_ascii=False))
        return 0
    for field, e in found.items():
        if e is None:
            print(f"{field:<6} not found")
            continue
        check = f"; {_CHECK[e.checked]}" if _CHECK[e.checked] else ""
        score = f"; score {e.score:.2f}" if e.score is not None else ""
        print(f"{field:<6} {e.value}  ({e.rule}, row {e.line + 1}: {e.evidence.strip()!r}{check}{score})")
    return 0


def _cmd_eval_extract(args: argparse.Namespace) -> int:
    config = OcrConfig(lang=args.lang, psm=args.psm)
    rows, sizes = [], {}
    for split in args.splits:
        receipts = load_split(Path(args.data), split)
        if not receipts:
            print(f"No receipts in the {split} split under {args.data}.", file=sys.stderr)
            continue
        print(f"{split}: {len(receipts)} receipts, OCR cached under {args.cache}...", file=sys.stderr)
        ocr = ocr_receipts(receipts, args.prep, config, Path(args.cache), args.workers)
        for r in receipts:
            rows += evaluate_extract.score_receipt(split, r, "ocr", ocr[r.id].text.splitlines())
            rows += evaluate_extract.score_receipt(split, r, "reference", r.reference_text.splitlines())
        sizes[split] = len(receipts)
    if not rows:
        return 2
    out = Path(args.out)
    evaluate_extract.write_csv(rows, out / "extract_eval.csv")
    header = (
        f"Rules from `folio/extract.py`. They were written looking only at the `dev` split "
        f"({sizes.get('dev', 0)} receipts, the stage 1 sample). The `test` split "
        f"({sizes.get('test', 0)} other receipts, drawn with seed 1) was scored once the rules were frozen. "
        f"OCR: `{args.prep}` preprocessing, page mode {args.psm}, "
        f"Tesseract {pytesseract.get_tesseract_version()}. *Reference* runs the same rules on the dataset's "
        "own transcription, to tell reading errors from rule errors. Values are compared after normalising, "
        "so `RM 5.90` equals `5,90`. A receipt whose label for a field is empty is left out of that field."
    )
    report = evaluate_extract.render_markdown(rows, header)
    (out / "extract_eval.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def _cmd_eval_learned(args: argparse.Namespace) -> int:
    config = OcrConfig(lang=args.lang, psm=args.psm)
    data = {}
    for split in ("dev", "train", "test"):
        receipts = load_split(Path(args.data), split)
        print(f"{split}: {len(receipts)} receipts, OCR cached under {args.cache}...", file=sys.stderr)
        ocr = ocr_receipts(receipts, args.prep, config, Path(args.cache), args.workers)
        data[split] = [(r, ocr[r.id]) for r in receipts]
    if not data["test"] or not (data["dev"] or data["train"]):
        print(f"Not enough receipts under {args.data}.", file=sys.stderr)
        return 2
    pool = [(r, o, split) for split in ("dev", "train") for r, o in data[split]]
    report, ranker = evaluate_learned.run(pool, data["test"])
    out = Path(args.out)
    evaluate_learned.write_csv(report.rows, out / "learned_eval.csv")
    header = (
        "A ranker scores every amount OCR found on a receipt and answers with the best one "
        "(`folio/learn.py`). It is fitted on the `dev` and `train` splits, chosen by cross-validation "
        f"there, and scored once on the `test` split, the same {len(report.rows)} labelled receipts the "
        "stage 2 rules were scored on. "
        f"OCR: `{args.prep}` preprocessing, page mode {args.psm}."
    )
    report_md = evaluate_learned.render_markdown(report, header)
    (out / "learned_eval.md").write_text(report_md, encoding="utf-8")
    if ranker.kind == "logistic":
        ranker.save(Path(args.model))
        print(f"Model saved to {args.model}", file=sys.stderr)
    print(report_md)
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

    p_ext = sub.add_parser("extract", help="read one receipt and print its total and date")
    p_ext.add_argument("image")
    p_ext.add_argument("--prep", default="gray", choices=sorted(PIPELINES))
    p_ext.add_argument("--psm", type=int, default=6)
    p_ext.add_argument("--lang", default="eng")
    p_ext.add_argument("--json", action="store_true", help="print JSON with the evidence for each field")
    p_ext.add_argument("--ranker", help="answer the total with a saved ranker, e.g. models/total_ranker.json")
    p_ext.set_defaults(func=_cmd_extract)

    p_eext = sub.add_parser("eval-extract", help="score the total and date extractors on SROIE")
    p_eext.add_argument("--data", default="data/sroie/data", help="folder with img/, box/ and key/")
    p_eext.add_argument("--splits", nargs="+", default=["dev", "test"], choices=SPLITS)
    p_eext.add_argument("--prep", default="gray", choices=sorted(PIPELINES))
    p_eext.add_argument("--psm", type=int, default=6)
    p_eext.add_argument("--lang", default="eng")
    p_eext.add_argument("--cache", default=str(DEFAULT_CACHE), help="where OCR output is kept between runs")
    p_eext.add_argument("--workers", type=int, default=2)
    p_eext.add_argument("--out", default="results")
    p_eext.set_defaults(func=_cmd_eval_extract)

    p_learn = sub.add_parser("eval-learned",
                             help="fit the total ranker, choose it by cross-validation, test it")
    p_learn.add_argument("--data", default="data/sroie/data", help="folder with img/, box/ and key/")
    p_learn.add_argument("--prep", default="gray", choices=sorted(PIPELINES))
    p_learn.add_argument("--psm", type=int, default=6)
    p_learn.add_argument("--lang", default="eng")
    p_learn.add_argument("--cache", default=str(DEFAULT_CACHE))
    p_learn.add_argument("--workers", type=int, default=2)
    p_learn.add_argument("--out", default="results")
    p_learn.add_argument("--model", default="models/total_ranker.json", help="where to save the fitted model")
    p_learn.set_defaults(func=_cmd_eval_learned)
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
