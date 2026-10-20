"""Command line: `folio ocr` and `folio extract` for one image, and one `folio eval-*`
command per experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import pytesseract

from . import evaluate_classify, evaluate_cluster, evaluate_extract, evaluate_learned, evaluate_llm
from .cache import DEFAULT_CACHE, ocr_receipts
from .classify import load_docs, split_docs
from .evaluate import compare, evaluate, render_markdown, summarize, write_csv
from .extract import EXTRACTORS
from .learn import TotalRanker
from .llm import (
    BACKENDS,
    SECOND_READING,
    CachedBackend,
    amounts_read,
    cascade,
    consult_image,
    consult_text,
    cost_usd,
)
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
    if args.ranker or args.llm:
        ranked = TotalRanker.load(Path(args.ranker or "models/total_ranker.json")).extract(result.rows)
        found["total"] = ranked
        if args.llm:
            backend = BACKENDS[args.llm](**({"model": args.llm_model} if args.llm_model else {}))
            if args.llm_input == "image":
                def consult():
                    return consult_image(backend.ask, Path(args.image).read_bytes(), lines)
            else:
                def consult():
                    return consult_text(backend.ask, lines)
            reading = None
            if args.guard:
                second = run_ocr(PIPELINES[SECOND_READING](img), OcrConfig(lang=args.lang, psm=args.psm))
                reading = amounts_read(second.text)
            routed = cascade(lines, EXTRACTORS["rules"]["total"](lines), found["date"], ranked, consult,
                             second_reading=reading)
            found["total"], found["date"] = routed.total, routed.date
            print(f"[route: {routed.route}]", file=sys.stderr)
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


def _cmd_eval_llm(args: argparse.Namespace) -> int:
    receipts = load_split(Path(args.data), args.split)
    if args.limit:
        receipts = receipts[: args.limit]
    if not receipts:
        print(f"No receipts in the {args.split} split under {args.data}.", file=sys.stderr)
        return 2
    ocr = ocr_receipts(receipts, args.prep, OcrConfig(lang=args.lang, psm=args.psm), Path(args.cache),
                       args.workers)
    ranker = TotalRanker.load(Path(args.ranker))
    options = {"model": args.model} if args.model else {}
    if args.backend == "anthropic":
        options["thinking"] = args.thinking
    backend = BACKENDS[args.backend](**options)
    cached = CachedBackend(backend, Path(args.llm_cache))
    asked = 0

    def ask(receipt_id: str, prompt: str, image=None, schema=None):
        nonlocal asked
        spent = cost_usd(backend.model, cached.new_input_tokens, cached.new_output_tokens)
        if spent is not None and spent > args.max_usd:
            raise SystemExit(f"Stopped: this run has spent about ${spent:.4f}, over --max-usd "
                             f"{args.max_usd}. Answers so far are cached, so rerunning resumes from here.")
        asked += 1
        print(f"  asking {backend.name} about receipt {receipt_id} ({asked})", file=sys.stderr)
        return cached.ask(receipt_id, prompt, image, schema)

    second = None
    if args.guard:
        second = ocr_receipts(receipts, SECOND_READING, OcrConfig(lang=args.lang, psm=args.psm),
                              Path(args.cache), args.workers)
    rows = evaluate_llm.evaluate([(r, ocr[r.id]) for r in receipts], ranker, ask, args.mode, args.threshold,
                                 args.input, second)
    tag = f"{args.split}_{args.mode}_{args.input}_{backend.name}_{backend.label}"
    if args.guard:
        tag += "_guard"
    tag = tag.replace(":", "-").replace("/", "-")
    out = Path(args.out)
    evaluate_llm.write_csv(rows, out / f"llm_eval_{tag}.csv")
    header = (
        f"Model `{backend.label}` through `{backend.name}` reading the "
        f"{'scan' if args.input == 'image' else 'OCR text'}, "
        f"mode `{args.mode}`, {len(rows)} labelled "
        "receipts "
        f"of the `{args.split}` split. OCR: `{args.prep}`, page mode {args.psm}. Ranker: `{args.ranker}`, "
        f"routing threshold {args.threshold}. Requests sent in this run: {cached.calls} (the rest came from "
        f"the cache in `{args.llm_cache}`)."
    )
    if args.guard:
        header += (f" Guard on: a trusted total must also be read by a second OCR pass (`{SECOND_READING}`, "
                   f"page mode {args.psm}), and the cash check does not confirm its own fallback.")
    tokens_in = sum(r.input_tokens or 0 for r in rows)
    tokens_out = sum(r.output_tokens or 0 for r in rows)
    total = cost_usd(backend.model, tokens_in, tokens_out)
    run = cost_usd(backend.model, cached.new_input_tokens, cached.new_output_tokens)
    if total is not None:
        header += (f" Cost at list prices: ${total:.4f} for all the answers in this report, "
                   f"${run:.4f} of it spent in this run.")
    report = evaluate_llm.render_markdown(rows, header, args.input)
    (out / f"llm_eval_{tag}.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def _cmd_eval_classify(args: argparse.Namespace) -> int:
    docs = load_docs(Path(args.data))
    if not docs:
        print(f"No documents under {args.data}. Run: bash scripts/get_tobacco_text.sh", file=sys.stderr)
        return 2
    split = split_docs(docs, args.test_share, args.seed)
    report = evaluate_classify.run(split["train"], split["test"], folds=args.folds, repeats=args.repeats,
                                   log=lambda message: print(message, file=sys.stderr))
    tag, cost = "", ""
    if args.llm:
        backend = BACKENDS[args.llm](**({"model": args.model} if args.model else {}))
        cached = CachedBackend(backend, Path(args.llm_cache))

        def ask(key, prompt, image=None, schema=None, system=None):
            spent = cost_usd(backend.model, cached.new_input_tokens, cached.new_output_tokens)
            if spent is not None and spent > args.max_usd:
                raise SystemExit(f"Stopped: this run has spent about ${spent:.4f}, over --max-usd "
                                 f"{args.max_usd}. Answers so far are cached: rerun to resume.")
            return cached.ask(key, prompt, image, schema, system)

        evaluate_classify.ask_llm(report.rows, split["test"], ask, args.mode)
        tag = f"_{args.mode}_{backend.name}_{backend.label}".replace(":", "-").replace("/", "-")
        asked = [r for r in report.rows if r.input_tokens]
        total = cost_usd(backend.model, sum(r.input_tokens or 0 for r in asked),
                         sum(r.output_tokens or 0 for r in asked))
        cost = (f"Model `{backend.label}` through `{backend.name}`, {cached.calls} requests sent in this run "
                f"(the rest came from the cache in `{args.llm_cache}`).")
        if total is not None:
            cost += f" Cost at list prices: ${total:.4f} for all the answers in this report."
    out = Path(args.out)
    evaluate_classify.write_csv(report.rows, out / f"classify_eval{tag}.csv")
    header = (
        f"Tesseract text of the Tobacco3482 documents (QS-OCR-Small, `{args.data}`): {len(docs)} documents "
        f"in {len({d.label for d in docs})} classes, split once with seed {args.seed} into {report.n_train} "
        f"for training and {report.n_test} held out for testing, the same share of every class in each."
    )
    md = evaluate_classify.render_markdown(report, header, cost)
    (out / f"classify_eval{tag}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _cmd_eval_cluster(args: argparse.Namespace) -> int:
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
    ranker = TotalRanker.load(Path(args.ranker)) if Path(args.ranker).exists() else None
    llm_ok = None
    if args.llm_report and Path(args.llm_report).exists():
        with Path(args.llm_report).open(encoding="utf-8") as f:
            llm_ok = {row["receipt"]: row["llm_total_ok"] == "True" for row in csv.DictReader(f)
                      if row["llm_total_ok"] in ("True", "False")}
    report = evaluate_cluster.run(data["dev"] + data["train"], data["test"], ranker, llm_ok,
                                  repeats=args.repeats, log=lambda message: print(message, file=sys.stderr))
    out = Path(args.out)
    evaluate_cluster.write_csv(report.rows, out / "cluster_eval.csv")
    header = (
        "Receipts are grouped from their OCR, never from their labels; the merchant names in the SROIE "
        "labels only score the groups (branches of a few chains and label typos merged, see "
        f"`folio/cluster.py`). OCR: `{args.prep}`, page mode {args.psm}. Ranker: `{args.ranker}`. "
        f"Model answers: `{args.llm_report}`."
    )
    md = evaluate_cluster.render_markdown(report, header)
    (out / "cluster_eval.md").write_text(md, encoding="utf-8")
    print(md)
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
    p_ext.add_argument("--llm", choices=sorted(BACKENDS), help="run the cascade, asking this LLM when unsure")
    p_ext.add_argument("--llm-model", help="model name for --llm, e.g. claude-haiku-5-5 or qwen2.5:3b")
    p_ext.add_argument("--llm-input", choices=["text", "image"], default="text",
                       help="the LLM reads the OCR text or the scan itself")
    p_ext.add_argument("--guard", action="store_true",
                       help="with --llm: distrust a cheap total that a second OCR pass does not read too")
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

    p_llm = sub.add_parser("eval-llm", help="score the LLM extractor and the cascade")
    p_llm.add_argument("--backend", choices=sorted(BACKENDS), default="ollama")
    p_llm.add_argument("--model", help="defaults: claude-haiku-5-5 for anthropic, qwen2.5:3b for ollama")
    p_llm.add_argument("--thinking", choices=["off", "low", "medium", "high"], default="off",
                       help="Claude only: no reasoning (off) or adaptive thinking at this effort")
    p_llm.add_argument("--input", choices=["text", "image"], default="text",
                       help="the model reads the OCR text (with row citations) or the scanned image")
    p_llm.add_argument("--mode", choices=["cascade", "all"], default="cascade",
                       help="ask the model only when unsure (cascade) or about every receipt (all)")
    p_llm.add_argument("--split", choices=SPLITS, default="test")
    p_llm.add_argument("--limit", type=int, help="only the first N receipts, for a quick try")
    p_llm.add_argument("--threshold", type=float, default=evaluate_learned.ROUTE_THRESHOLD)
    p_llm.add_argument("--guard", action="store_true",
                       help="distrust a cheap total that a second OCR pass does not read too")
    p_llm.add_argument("--max-usd", type=float, default=1.0,
                       help="stop sending requests once this run has spent about this much (API models only)")
    p_llm.add_argument("--ranker", default="models/total_ranker.json")
    p_llm.add_argument("--data", default="data/sroie/data", help="folder with img/, box/ and key/")
    p_llm.add_argument("--prep", default="gray", choices=sorted(PIPELINES))
    p_llm.add_argument("--psm", type=int, default=6)
    p_llm.add_argument("--lang", default="eng")
    p_llm.add_argument("--cache", default=str(DEFAULT_CACHE))
    p_llm.add_argument("--llm-cache", default="cache/llm")
    p_llm.add_argument("--workers", type=int, default=2)
    p_llm.add_argument("--out", default="results")
    p_llm.set_defaults(func=_cmd_eval_llm)

    p_cls = sub.add_parser("eval-classify", help="compare document-type classifiers on Tobacco3482 OCR text")
    p_cls.add_argument("--data", default="data/qs-ocr-small", help="one folder of .txt files per class")
    p_cls.add_argument("--test-share", type=float, default=0.2)
    p_cls.add_argument("--seed", type=int, default=0)
    p_cls.add_argument("--folds", type=int, default=5)
    p_cls.add_argument("--repeats", type=int, default=2)
    p_cls.add_argument("--llm", choices=sorted(BACKENDS), help="also ask this LLM for the document type")
    p_cls.add_argument("--model", help="defaults: claude-haiku-5-5 for anthropic, qwen2.5:3b for ollama")
    p_cls.add_argument("--mode", choices=["cascade", "all"], default="cascade",
                       help="ask the LLM only about the documents the classifier is unsure of, or about all")
    p_cls.add_argument("--max-usd", type=float, default=0.5)
    p_cls.add_argument("--llm-cache", default="cache/llm-classify")
    p_cls.add_argument("--out", default="results")
    p_cls.set_defaults(func=_cmd_eval_classify, needs_tesseract=False)

    p_clu = sub.add_parser("eval-cluster", help="group SROIE receipts by layout without labels")
    p_clu.add_argument("--data", default="data/sroie/data", help="folder with img/, box/ and key/")
    p_clu.add_argument("--prep", default="gray", choices=sorted(PIPELINES))
    p_clu.add_argument("--psm", type=int, default=6)
    p_clu.add_argument("--lang", default="eng")
    p_clu.add_argument("--cache", default=str(DEFAULT_CACHE))
    p_clu.add_argument("--workers", type=int, default=2)
    p_clu.add_argument("--ranker", default="models/total_ranker.json")
    p_clu.add_argument("--llm-report", help="stage 3 answers of the model reading the scan",
                       default="results/llm_eval_test_all_image_anthropic_claude-haiku-5-5.csv")
    p_clu.add_argument("--repeats", type=int, default=3)
    p_clu.add_argument("--out", default="results")
    p_clu.set_defaults(func=_cmd_eval_cluster)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "needs_tesseract", True) and not tesseract_available():
        print("Tesseract is not installed. On Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-spa",
              file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
