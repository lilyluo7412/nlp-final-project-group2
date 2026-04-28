from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer


def score_tfidf(doc_term_counts: dict[str, Counter[str]]) -> list[tuple[str, float]]:
    doc_ids = sorted(doc_term_counts.keys())
    term_to_idx: dict[str, int] = {}
    for d in doc_ids:
        for t in doc_term_counts[d]:
            if t not in term_to_idx:
                term_to_idx[t] = len(term_to_idx)
    if not term_to_idx:
        return []

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, d in enumerate(doc_ids):
        for term, cnt in doc_term_counts[d].items():
            rows.append(i)
            cols.append(term_to_idx[term])
            data.append(float(cnt))

    X = csr_matrix((data, (rows, cols)), shape=(len(doc_ids), len(term_to_idx)))
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    Xt = tfidf.fit_transform(X)
    col_sums = np.asarray(Xt.sum(axis=0)).ravel()
    idx_to_term = [None] * len(term_to_idx)
    for t, j in term_to_idx.items():
        idx_to_term[j] = t
    ranked = [(idx_to_term[j], float(col_sums[j])) for j in range(len(idx_to_term))]
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked
