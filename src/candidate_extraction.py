from __future__ import annotations

from collections import Counter

from .preprocessing import normalize_term

ADJ_TAGS = {"JJ", "JJR", "JJS"}
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}

BAD_SPAN_START_WORDS = frozenset({
    "much", "own", "little", "same", "whole", "such",
})

def _matches_pattern(tags: list[str], min_len: int, max_len: int) -> bool:
    if all(t in {"NNP", "NNPS"} for t in tags):
        return False
    n = len(tags)
    if n < min_len or n > max_len:
        return False
    for split_idx in range(0, n):
        if split_idx > 0 and not all(t in ADJ_TAGS for t in tags[:split_idx]):
            continue
        noun_part = tags[split_idx:]
        if noun_part and all(t in NOUN_TAGS for t in noun_part):
            return True
    return False

def extract_candidates_per_sentence(
    tagged_sentences: list[list[tuple[str, str]]],
    min_len: int = 2,
    max_len: int = 4,
) -> list[list[str]]:
    """Returns candidates grouped by sentence (used for co-occurrence scoring)."""
    per_sentence: list[list[str]] = []
    for tagged in tagged_sentences:
        words = [w for w, _ in tagged]
        tags = [t for _, t in tagged]
        n = len(words)
        sent_candidates: list[str] = []
        for i in range(n):
            for ln in range(min_len, min(max_len, n - i) + 1):
                tseg = tags[i : i + ln]
                if not _matches_pattern(tseg, min_len=min_len, max_len=max_len):
                    continue
                # skip spans starting with bad words
                if words[i].lower() in BAD_SPAN_START_WORDS:
                    continue
                span = [normalize_term(x) for x in words[i : i + ln]]
                span = [x for x in span if x]
                if len(span) != ln:
                    continue
                sent_candidates.append(" ".join(span))
        if sent_candidates:
            per_sentence.append(sent_candidates)
    return per_sentence

def extract_candidates_from_tagged_sentences(
    tagged_sentences: list[list[tuple[str, str]]],
    min_len: int = 2,
    max_len: int = 4,
) -> list[str]:
    candidates: list[str] = []
    for tagged in tagged_sentences:
        words = [w for w, _ in tagged]
        tags = [t for _, t in tagged]
        n = len(words)
        for i in range(n):
            for ln in range(min_len, min(max_len, n - i) + 1):
                tseg = tags[i : i + ln]
                if not _matches_pattern(tseg, min_len=min_len, max_len=max_len):
                    continue
                # skip spans starting with bad words
                if words[i].lower() in BAD_SPAN_START_WORDS:
                    continue
                span = [normalize_term(x) for x in words[i : i + ln]]
                span = [x for x in span if x]
                if len(span) != ln:
                    continue
                candidates.append(" ".join(span))
    return candidates


def build_doc_candidate_counts(
    doc_to_candidates: dict[str, list[str]],
    min_freq: int,
) -> dict[str, Counter[str]]:
    global_counts = Counter()
    for cands in doc_to_candidates.values():
        global_counts.update(cands)
    out: dict[str, Counter[str]] = {}
    for doc_id, cands in doc_to_candidates.items():
        out[doc_id] = Counter([c for c in cands if global_counts[c] >= min_freq])
    return out
