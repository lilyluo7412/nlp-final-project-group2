from __future__ import annotations

from collections import Counter, defaultdict


def _build_term_windows(term_sequences: list[list[str]], window_size: int) -> dict[tuple[str, str], int]:
    cooc: dict[tuple[str, str], int] = defaultdict(int)
    w = max(2, window_size)
    for seq in term_sequences:
        n = len(seq)
        for i in range(n):
            for j in range(i + 1, min(n, i + w)):
                a, b = seq[i], seq[j]
                if a == b:
                    continue
                pair = (a, b) if a <= b else (b, a)
                cooc[pair] += 1
    return dict(cooc)


def score_cooccurrence_original(
    doc_term_counts: dict[str, Counter[str]],
    window_size: int = 5,
) -> list[tuple[str, float]]:
    # Keep simple: create term streams per document, score each term by
    # sum of co-occurrence counts with other terms within a fixed window.
    term_sequences: list[list[str]] = []
    term_totals: Counter[str] = Counter()
    for _, counter in doc_term_counts.items():
        seq: list[str] = []
        for term, cnt in counter.items():
            seq.extend([term] * int(cnt))
            term_totals[term] += cnt
        term_sequences.append(seq)

    cooc = _build_term_windows(term_sequences, window_size=window_size)
    scores: dict[str, float] = {t: 0.0 for t in term_totals}
    for (a, b), c in cooc.items():
        scores[a] += float(c)
        scores[b] += float(c)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked

def score_cooccurrence(
    doc_sent_candidates: dict[str, list[list[str]]],
    window_size: int = 5,
) -> list[tuple[str, float]]:
    """
    Fixed implementation. Takes sentence-level candidate sequences per document.
    Co-occurrence is measured within actual sentence windows, not frequency-expanded lists.
    """
    term_totals: Counter[str] = Counter()
    all_sequences: list[list[str]] = []
 
    for sent_lists in doc_sent_candidates.values():
        for sent_candidates in sent_lists:
            if sent_candidates:
                all_sequences.append(sent_candidates)
                term_totals.update(sent_candidates)
 
    cooc = _build_term_windows(all_sequences, window_size=window_size)
    scores: dict[str, float] = {t: 0.0 for t in term_totals}
    for (a, b), c in cooc.items():
        scores[a] += float(c)
        scores[b] += float(c)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked
 
