#!/usr/bin/env python3
"""
Wire gold TSV + split lists for the frozen pipeline (run from repo root).

  python3 scripts/build_wired_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _token_count(term: str) -> int:
    return len(term.split())


def _normalize_ws(s: str) -> str:
    return " ".join(s.split())


def read_tsv(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return []
    header = lines[0].lower()
    if "\t" not in header:
        raise ValueError(f"Expected tab-separated TSV: {path}")
    rows: list[tuple[str, str]] = []
    for i, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        doc_id, term = parts[0].strip(), "\t".join(parts[1:]).strip()
        if doc_id and term:
            rows.append((doc_id, term))
    return rows


def group_and_filter(
    rows: list[tuple[str, str]],
    label: str,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """
    Returns (doc_id -> ordered unique terms), stats.
    """
    stats = {
        "input_rows": len(rows),
        "dropped_duplicate_within_doc": 0,
        "dropped_not_2_to_4_tokens": 0,
    }
    buckets: dict[str, list[str]] = {}
    seen_per_doc: dict[str, set[str]] = {}

    for doc_id, term in rows:
        t = _normalize_ws(term)
        key = t.lower()
        if doc_id not in seen_per_doc:
            seen_per_doc[doc_id] = set()
            buckets[doc_id] = []
        if key in seen_per_doc[doc_id]:
            stats["dropped_duplicate_within_doc"] += 1
            continue
        n = _token_count(t)
        if n < 2 or n > 4:
            stats["dropped_not_2_to_4_tokens"] += 1
            continue
        seen_per_doc[doc_id].add(key)
        buckets[doc_id].append(t)

    return buckets, stats


def write_jsonl(path: Path, doc_to_terms: dict[str, list[str]]) -> int:
    lines: list[str] = []
    total = 0
    for doc_id in sorted(doc_to_terms.keys()):
        terms = doc_to_terms[doc_id]
        total += len(terms)
        lines.append(json.dumps({"doc_id": doc_id, "terms": terms}, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return total


def stems_from_dir(d: Path) -> list[str]:
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.txt"))


def main() -> None:
    raw_dev = REPO_ROOT / "data" / "raw" / "dev"
    raw_test = REPO_ROOT / "data" / "raw" / "test"
    dev_stems = stems_from_dir(raw_dev)
    test_stems = stems_from_dir(raw_test)

    (REPO_ROOT / "data" / "splits" / "dev_ids.txt").write_text(
        "\n".join(dev_stems) + ("\n" if dev_stems else ""),
        encoding="utf-8",
    )
    (REPO_ROOT / "data" / "splits" / "test_ids.txt").write_text(
        "\n".join(test_stems) + ("\n" if test_stems else ""),
        encoding="utf-8",
    )

    dev_rows = read_tsv(REPO_ROOT / "data" / "gold_standard" / "gold_standard_dev.tsv")
    test_rows = read_tsv(REPO_ROOT / "data" / "gold_standard" / "gold_standard_test.tsv")

    dev_buckets, dev_stats = group_and_filter(dev_rows, "dev")
    test_buckets, test_stats = group_and_filter(test_rows, "test")

    total_dev_terms = write_jsonl(REPO_ROOT / "data" / "annotations" / "gold_terms_dev.jsonl", dev_buckets)
    total_test_terms = write_jsonl(REPO_ROOT / "data" / "annotations" / "gold_terms_test.jsonl", test_buckets)

    report_lines = [
        "Gold TSV -> JSONL conversion report",
        "",
        f"Raw dev dir: {raw_dev} ({len(dev_stems)} .txt files)",
        f"Raw test dir: {raw_test} ({len(test_stems)} .txt files)",
        "",
        "## DEV (gold_standard_dev.tsv)",
        f"  input_rows: {dev_stats['input_rows']}",
        f"  dropped_duplicate_within_doc: {dev_stats['dropped_duplicate_within_doc']}",
        f"  dropped_not_2_to_4_tokens (unigrams, >4 words, etc.): {dev_stats['dropped_not_2_to_4_tokens']}",
        f"  retained_terms (sum over docs): {total_dev_terms}",
        "",
        "## TEST (gold_standard_test.tsv)",
        f"  input_rows: {test_stats['input_rows']}",
        f"  dropped_duplicate_within_doc: {test_stats['dropped_duplicate_within_doc']}",
        f"  dropped_not_2_to_4_tokens (unigrams, >4 words, etc.): {test_stats['dropped_not_2_to_4_tokens']}",
        f"  retained_terms (sum over docs): {total_test_terms}",
        "",
        "Note: token count = whitespace-separated words (e.g. COBE, WMAP dropped as 1 token).",
        "",
        "Wrote:",
        "  data/splits/dev_ids.txt",
        "  data/splits/test_ids.txt",
        "  data/annotations/gold_terms_dev.jsonl",
        "  data/annotations/gold_terms_test.jsonl",
    ]
    report_path = REPO_ROOT / "data" / "annotations" / "gold_conversion_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
