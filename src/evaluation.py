from __future__ import annotations

import json
from pathlib import Path

from .preprocessing import normalize_term

KS = (10, 20, 50)


def load_gold_jsonl(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        doc_id = obj["doc_id"]
        terms = obj.get("terms", [])
        out[doc_id] = {normalize_term(t) for t in terms if normalize_term(t)}
    return out


def precision_at_k(ranked_terms: list[str], gold_terms: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = ranked_terms[:k]
    hits = sum(1 for t in topk if normalize_term(t) in gold_terms)
    return hits / float(k)


def evaluate_precision(ranked_terms: list[str], gold_terms: set[str]) -> dict[str, float]:
    return {f"P@{k}": precision_at_k(ranked_terms, gold_terms, k) for k in KS}
