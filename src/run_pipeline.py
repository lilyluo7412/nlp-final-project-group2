from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .candidate_extraction import build_doc_candidate_counts, extract_candidates_from_tagged_sentences, extract_candidates_per_sentence
from .evaluation import evaluate_precision, load_gold_jsonl
from .preprocessing import load_documents, tokenize_and_tag
from .scoring_cooccurrence import score_cooccurrence
from .scoring_tfidf import score_tfidf


def _read_ids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_ranked_csv(path: Path, ranked: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "term", "score"])
        for i, (term, score) in enumerate(ranked, 1):
            w.writerow([i, term, f"{score:.6f}"])


def _write_eval_csv(path: Path, tfidf_metrics: dict[str, float], cooc_metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "P@10", "P@20", "P@50"])
        w.writerow(["tfidf", tfidf_metrics["P@10"], tfidf_metrics["P@20"], tfidf_metrics["P@50"]])
        w.writerow(["cooccurrence", cooc_metrics["P@10"], cooc_metrics["P@20"], cooc_metrics["P@50"]])


def run(args: argparse.Namespace) -> None:
    config = _load_simple_yaml(Path(args.config))
    min_freq = int(config["min_freq"])
    if getattr(args, "min_freq_override", None) is not None:
        min_freq = int(args.min_freq_override)
    min_len = int(config["min_len"])
    max_len = int(config["max_len"])
    window_size = int(config["window_size"])

    docs = load_documents(Path(args.raw_docs))
    dev_ids = _read_ids(Path(args.dev_ids))
    test_ids = _read_ids(Path(args.test_ids))
    dev_set = {d: docs[d] for d in dev_ids if d in docs}
    test_set = {d: docs[d] for d in test_ids if d in docs}

    # Candidate extraction on all docs in each split (no unigrams; strict 2-4).
    def build_counts(split_docs: dict[str, str]):
        extracted: dict[str, list[str]] = {}
        extracted_sent: dict[str, list[list[str]]] = {}
        for doc_id, text in split_docs.items():
            tagged = tokenize_and_tag(text)
            extracted[doc_id] = extract_candidates_from_tagged_sentences(
                tagged, min_len=min_len, max_len=max_len
            )
            extracted_sent[doc_id] = extract_candidates_per_sentence(
                tagged, min_len=min_len, max_len=max_len
            )
        return build_doc_candidate_counts(extracted, min_freq=min_freq), extracted_sent

    dev_counts, dev_sent = build_counts(dev_set)
    test_counts, test_sent = build_counts(test_set)

    # "Tuning happens on dev only": we record dev metrics only.
    dev_gold = load_gold_jsonl(Path(args.gold_dev))
    dev_gold_terms = set().union(*[dev_gold.get(doc_id, set()) for doc_id in dev_counts])
    dev_tfidf_ranked = score_tfidf(dev_counts)
    dev_cooc_ranked = score_cooccurrence(dev_sent, window_size=window_size)
    dev_tfidf_terms = [t for t, _ in dev_tfidf_ranked]
    dev_cooc_terms = [t for t, _ in dev_cooc_ranked]
    dev_metrics = {
        "tfidf": evaluate_precision(dev_tfidf_terms, dev_gold_terms),
        "cooccurrence": evaluate_precision(dev_cooc_terms, dev_gold_terms),
    }

    Path(args.outputs_dev).mkdir(parents=True, exist_ok=True)
    _write_ranked_csv(Path(args.outputs_dev) / "tfidf_ranked_terms.csv", dev_tfidf_ranked)
    _write_ranked_csv(Path(args.outputs_dev) / "cooc_ranked_terms.csv", dev_cooc_ranked)
    _write_eval_csv(
        Path(args.outputs_dev) / "precision_at_k_dev.csv",
        dev_metrics["tfidf"],
        dev_metrics["cooccurrence"],
    )
    (Path(args.outputs_dev) / "tuning_log.json").write_text(
        json.dumps({"config": config, "dev_metrics": dev_metrics}, indent=2), encoding="utf-8"
    )

    # Final test evaluation: use frozen config once.
    test_gold = load_gold_jsonl(Path(args.gold_test))
    test_gold_terms = set().union(*[test_gold.get(doc_id, set()) for doc_id in test_counts])
    test_tfidf_ranked = score_tfidf(test_counts)
    test_cooc_ranked = score_cooccurrence(test_sent, window_size=window_size)
    test_tfidf_terms = [t for t, _ in test_tfidf_ranked]
    test_cooc_terms = [t for t, _ in test_cooc_ranked]
    test_metrics_tfidf = evaluate_precision(test_tfidf_terms, test_gold_terms)
    test_metrics_cooc = evaluate_precision(test_cooc_terms, test_gold_terms)

    Path(args.outputs_test).mkdir(parents=True, exist_ok=True)
    _write_ranked_csv(Path(args.outputs_test) / "tfidf_ranked_terms.csv", test_tfidf_ranked)
    _write_ranked_csv(Path(args.outputs_test) / "cooc_ranked_terms.csv", test_cooc_ranked)
    _write_eval_csv(
        Path(args.outputs_test) / "precision_at_k_test.csv",
        test_metrics_tfidf,
        test_metrics_cooc,
    )

    print("Done. Outputs written to:")
    print(f"- {args.outputs_dev}")
    print(f"- {args.outputs_test}")


def _load_simple_yaml(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        out[k.strip()] = int(v.strip())
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Terminology extraction pipeline (frozen design).")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--raw-docs", default="data/raw")
    p.add_argument("--dev-ids", default="data/splits/dev_ids.txt")
    p.add_argument("--test-ids", default="data/splits/test_ids.txt")
    p.add_argument("--gold-dev", default="data/annotations/gold_terms_dev.jsonl")
    p.add_argument("--gold-test", default="data/annotations/gold_terms_test.jsonl")
    p.add_argument("--outputs-dev", default="outputs/dev")
    p.add_argument("--outputs-test", default="outputs/test")
    p.add_argument(
        "--min-freq",
        type=int,
        default=None,
        dest="min_freq_override",
        help="Optional: override config.yaml min_freq (e.g., dev-only ablations without editing the file).",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()