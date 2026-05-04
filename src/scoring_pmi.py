from __future__ import annotations

import math
from collections import Counter, defaultdict


def score_pmi(
    doc_sent_candidates: dict[str, list[list[str]]],
    window_size: int = 5,
    min_pair_count: int = 7,
) -> list[tuple[str, float]]:
    """
    PPMI-based term scoring (Positive PMI weighted by frequency).

    Uses PPMI = max(0, PMI) weighted by log(joint_count) to avoid
    artificially high scores for rare co-occurrences.

    All three probabilities use consistent denominators:
    - P(a) = count(a) / total_terms  (unigram probability)
    - P(b) = count(b) / total_terms  (unigram probability)
    - P(a,b) = count(a,b) / total_windows  (pair probability)

    where total_windows = total number of (i,j) pairs observed across all sequences.

    score(a) = mean of PPMI(a,b) * log(count(a,b)+1)
               over all b where count(a,b) >= min_pair_count
    """
    term_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    all_sequences: list[list[str]] = []
    total_terms = 0

    for sent_lists in doc_sent_candidates.values():
        for sent_candidates in sent_lists:
            if not sent_candidates:
                continue
            all_sequences.append(sent_candidates)
            term_counts.update(sent_candidates)
            total_terms += len(sent_candidates)

    if total_terms == 0:
        return []

    # Step 2: count co-occurring pairs and total windows
    w = max(2, window_size)
    total_windows = 0
    for seq in all_sequences:
        n = len(seq)
        for i in range(n):
            for j in range(i + 1, min(n, i + w)):
                a, b = seq[i], seq[j]
                if a == b:
                    continue
                pair = (a, b) if a <= b else (b, a)
                pair_counts[pair] += 1
                total_windows += 1  # each (i,j) pair is one co-occurrence opportunity

    if total_windows == 0:
        return []

    # Step 3: compute PPMI with consistent probability space
    term_pmi_scores: dict[str, list[float]] = defaultdict(list)

    for (a, b), joint_count in pair_counts.items():
        if joint_count < min_pair_count:
            continue

        # P(a,b) uses total_windows as denominator
        p_ab = joint_count / total_windows
        # P(a) and P(b) use total_terms as denominator (unigram probabilities)
        # count term appearances within windows (consistent with p_ab)
        window_term_counts: Counter[str] = Counter()
        for seq in all_sequences:
            n = len(seq)
            for i in range(n):
                for j in range(i + 1, min(n, i + w)):
                    window_term_counts[seq[i]] += 1
                    window_term_counts[seq[j]] += 1

        total_window_terms = max(1, sum(window_term_counts.values()))

        # then in PMI computation:
        p_a = window_term_counts[a] / total_window_terms
        p_b = window_term_counts[b] / total_window_terms
        p_ab = joint_count / total_windows

        if p_a > 0 and p_b > 0 and p_ab > 0:
            pmi = math.log2(p_ab / (p_a * p_b))
            # PPMI: clip to 0, weight by log frequency to discount rare pairs
            weighted_pmi = max(0.0, pmi) * math.log(joint_count + 1)
            term_pmi_scores[a].append(weighted_pmi)
            term_pmi_scores[b].append(weighted_pmi)

    # Step 4: score each term by mean weighted PPMI
    scores: dict[str, float] = {}
    for term in term_counts:
        if term in term_pmi_scores and term_pmi_scores[term]:
            scores[term] = sum(term_pmi_scores[term]) / len(term_pmi_scores[term])
        else:
            scores[term] = 0.0

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked