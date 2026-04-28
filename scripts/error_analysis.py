#!/usr/bin/env python3
"""
Dev-only error analysis helper: top-k false positives and missed gold terms.
Uses the same normalized matching as src/evaluation.py.

  python3 scripts/error_analysis.py --split dev --k 10
  python3 scripts/error_analysis.py --split test --k 10
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.evaluation import load_gold_jsonl  # noqa: E402
from src.preprocessing import load_documents, normalize_term  # noqa: E402


def read_ids(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def gold_set_for_split(split: str) -> set[str]:
    """Match run_pipeline.py: merged gold only for doc_ids present in raw corpus."""
    ids_path = REPO / "data" / "splits" / ("dev_ids.txt" if split == "dev" else "test_ids.txt")
    gold_path = REPO / "data" / "annotations" / (
        "gold_terms_dev.jsonl" if split == "dev" else "gold_terms_test.jsonl"
    )
    docs = load_documents(REPO / "data" / "raw")
    want = read_ids(ids_path)
    split_docs = {d: docs[d] for d in want if d in docs}
    per_doc = load_gold_jsonl(gold_path)
    terms: set[str] = set()
    for doc_id in split_docs:
        terms |= per_doc.get(doc_id, set())
    return terms


def read_ranked_terms(csv_path: Path) -> list[str]:
    rows: list[str] = []
    with csv_path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row["term"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Error analysis vs merged gold for a split.")
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.split
    gold = gold_set_for_split(args.split)

    for method in ("tfidf", "cooccurrence"):
        stem = "tfidf_ranked_terms.csv" if method == "tfidf" else "cooc_ranked_terms.csv"
        path = out_dir / stem
        if not path.is_file():
            print(f"Missing {path}; run terminology_extraction.py first.")
            continue
        ranked = read_ranked_terms(path)
        k = max(1, args.k)
        topk = ranked[:k]
        topk_norm = {normalize_term(t) for t in topk}
        false_positives = [t for t in topk if normalize_term(t) not in gold]
        missed = sorted(g for g in gold if g not in topk_norm)

        print(f"=== {args.split.upper()} | {method} | top-{k} ===")
        print(f"False positives (predicted in top-{k}, not in gold norm), up to {k}:")
        for t in false_positives:
            print(f"  - {t}")
        print(f"Missed gold (in gold union, absent from top-{k} preds), n={len(missed)}:")
        for t in missed[:50]:
            print(f"  - {t}")
        if len(missed) > 50:
            print(f"  ... and {len(missed) - 50} more")
        print()


if __name__ == "__main__":
    main()
