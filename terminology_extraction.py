"""
NLP final project — domain-specific multi-word term extraction: TF-IDF baseline and co-occurrence.

- Candidates: **exact** contiguous 2–4 word phrases (POS: compound NNs or (Mod)*(Noun)+).
- A **base** filter removes stopword edges, verbal tails, etc.; a **quality** filter
  drops generic phrases (e.g. EXACT_LOW_QUALITY) and short prefix/suffix **fragments**
  of longer candidates (unless the phrase is in GOLD_TERMS or PROTECTED_TERMS).
- TF-IDF: one sentence = one document. Co-occurrence: sliding window over word tokens.
- Evaluation: **original** = lowercase + strip punct; **normalized** = noun lemmatization
  (singular/plural alignment). Paper output: `final_results.txt`.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer

# --- Editable gold annotations (replace/extend with your labeled terms) --------

GOLD_TERMS: list[str] = [
    "solar system",
    "planetary bodies",
    "planetary body",
    "terrestrial planets",
    "giant planets",
    "impact craters",
    "mean density",
    "atmospheric pressure",
    "greenhouse effect",
    "interstellar medium",
    "gravitational waves",
    "black holes",
    "space telescopes",
    "near side",
    "far side",
]

# Phrase length (words). Only exact lengths in this range are kept.
MIN_SPAN_WORDS = 2
MAX_SPAN_WORDS = 4

# Minimum times a phrase must appear in the **whole corpus** to pass filtering (1 = off).
MIN_GLOBAL_FREQ = 1
MIN_TOKEN_LEN = 2

# Default text file: place your textbook pages here
CORPUS_FILE = "corpus.txt"

# Evaluation and reporting
K_DEFAULTS = (10, 20, 50)

# Never drop these in the **quality** pass (lowercase, exact after preprocess).
PROTECTED_TERMS: frozenset[str] = frozenset(
    {
        "atmospheric pressure",
        "greenhouse effect",
        "orbital period",
        "rotation period",
        "impact craters",
        "solar system",
    }
)

# Weak / overly generic exact phrases to drop (in addition to subfragment rules).
EXACT_LOW_QUALITY: frozenset[str] = frozenset(
    {
        "formation process",
        "planetary formation",
        "planetary formation process",  # broad / head noun “process”; noisy for term ranking
        "long timescales",
        "long time",
    }
)

# Trailing words often mis-tagged or verbal — drop spans ending in these.
BAD_TAIL_TOKENS = frozenset(
    {
        "forms",
        "consists",
        "using",
        "including",
        "becoming",
    }
)

# Last word of a candidate is often a verb mis-tagged as NN/NNS (POS noise).
VERB_LIKE_TAIL_WORDS = frozenset(
    {
        "influences",
        "influence",
        "varies",
        "vary",
        "relate",
        "relates",
        "differ",
        "differs",
        "includes",
        "include",
        "appear",
        "appears",
        "depends",
        "depend",
        "occurs",
        "occur",
        "happens",
        "happen",
    }
)

# Reject a span *starting* at this token (POS over-generation: VB... / MD, or
# words often tagged NN but that head bad bigrams, e.g. "study" as common noun).
VERB_START_TAGS = frozenset({"VB", "VBD", "VBP", "VBZ", "VBG", "MD"})
# Reject any span containing a finite / lexical verb tag (keeps chunks inside one NP).
VERB_INSIDE_SPAN_BAD = frozenset({"VB", "VBD", "VBP", "VBZ", "VBG", "MD"})
# Do not start a candidate at this token (lowercase)
BAD_SPAN_START_WORDS = frozenset(
    {
        "study",
        "affect",  # often VB before object NP
    }
)

# --- NLTK ---------------------------------------------------------------------

try:
    import nltk
    from nltk import pos_tag
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import sent_tokenize, word_tokenize
except ImportError as e:  # pragma: no cover
    raise ImportError("Install nltk: pip install nltk") from e

_lemmatizer: WordNetLemmatizer | None = None


def ensure_nltk_data() -> None:
    for resource, download_name in (
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(download_name, quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


def get_stopwords() -> set[str]:
    ensure_nltk_data()
    try:
        from nltk.corpus import stopwords as nltk_stopwords

        return set(nltk_stopwords.words("english"))
    except Exception:
        return {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "as",
            "by",
            "with",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "it",
            "this",
            "that",
            "these",
            "those",
        }


def get_lemmatizer() -> WordNetLemmatizer:
    global _lemmatizer
    ensure_nltk_data()
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


# --- Preprocessing ------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_STRIP = re.compile(r"^[^a-z0-9]+|[^a-z0-9]+$", re.IGNORECASE)


def preprocess_text(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


def load_corpus_text(path: Path) -> str:
    """Read UTF-8 text from one file; strip BOM."""
    return path.read_text(encoding="utf-8").strip().lstrip("\ufeff")


def tokenize_sentences(text: str) -> list[str]:
    ensure_nltk_data()
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


# --- POS sets for exact spans: compound NN | (MOD)* (NOUN)+ -------------------

ADJ_TAGS = frozenset({"JJ", "JJR", "JJS"})
NOUN_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS", "FW"})
# Modifiers before a noun: adjectives, some adverbs (near/far), participles used as modifiers.
# RB/RBR/RBS: e.g. "far side" (RB + NN) when tagger returns RB, not IN.
MOD_PREFIX_TAGS = ADJ_TAGS | {"RB", "RBR", "RBS", "VBN"}


def _span_contains_verbal(tags: list[str]) -> bool:
    """Block 'surface environment influences' — if *influences* is VBZ, drop the span."""
    return any(t in VERB_INSIDE_SPAN_BAD for t in tags)


def _valid_np_span_pos(tags: list[str], min_w: int, max_w: int) -> bool:
    """
    True if tags describe a 2–4 word NP-like span:
    - 2+ consecutive nouns (noun + noun, incl. 'impact crater', 'space telescope')
    - or (optional MOD)* + (NOUN)+ with at least one noun and length in [min_w, max_w]
    (covers JJ+NN, JJ+JJ+NN, RB+JJ+NN, hyphenated 'ground-based' as one JJ token, etc.)
    """
    w = len(tags)
    if w < min_w or w > max_w:
        return False
    if all(t in NOUN_TAGS for t in tags):
        return True
    for k in range(0, w):
        if k > 0 and not all(tags[j] in MOD_PREFIX_TAGS for j in range(k)):
            continue
        rest = w - k
        if rest < 1:
            continue
        if not all(tags[j] in NOUN_TAGS for j in range(k, w)):
            continue
        return True
    return False


def _clean_tok(s: str) -> str:
    t = re.sub(r"^['\"]+|['\"]+$", "", s)
    return t


def _phrase_from_window(words: list[str], i: int, n: int) -> str:
    """Exact contiguous token phrase (lowercased) — must match corpus word sequence."""
    span = words[i : i + n]
    parts = [preprocess_text(_clean_tok(w)) for w in span]
    parts = [p for p in parts if p]
    if len(parts) != n:
        return ""
    return " ".join(parts)


def extract_exact_spans_from_sentence(
    sentence: str,
    min_w: int = MIN_SPAN_WORDS,
    max_w: int = MAX_SPAN_WORDS,
) -> list[str]:
    """
    All exact 2–4 word substrings of word_tokenize(sentence) whose POS window matches
    (Adj|RB|VBN)* (Noun)+ or (Noun){2+}. No made-up text — only token joins.
    """
    words = word_tokenize(sentence)
    if len(words) < min_w:
        return []
    tagged = pos_tag(words)
    tags = [t for _, t in tagged]
    n_tok = len(words)
    out: list[str] = []
    for i in range(n_tok):
        t0 = tags[i]
        if t0 in VERB_START_TAGS:
            continue
        w0 = _clean_tok(words[i]).lower()
        if w0 in BAD_SPAN_START_WORDS:
            continue
        for ln in range(min_w, min(max_w, n_tok - i) + 1):
            tseg = tags[i : i + ln]
            ok = _valid_np_span_pos(tseg, min_w, max_w)
            # e.g. "near side" / "far side" are often IN/RP + NN — keep exact two-word text
            if not ok and ln == 2:
                w0 = _clean_tok(words[i]).lower()
                if w0 in ("near", "far") and tags[i + 1] in NOUN_TAGS:
                    ok = True
            if ok and _span_contains_verbal(tseg):
                ok = False
            if not ok:
                continue
            p = _phrase_from_window(words, i, ln)
            if p:
                out.append(p)
    return out


# --- Light post-filters (avoid over-pruning) ----------------------------------

GENERIC_ADJECTIVES = frozenset(
    {
        "small",
        "smaller",
        "smallest",
        "large",
        "larger",
        "largest",
        "big",
        "bigger",
        "new",
        "old",
        "other",
        "different",
        "same",
        "certain",
        "many",
        "few",
        "solid",
    }
)
GENERIC_HEAD_NOUNS = frozenset(
    {
        "thing",
        "things",
        "way",
        "ways",
        "part",
        "parts",
        "kind",
        "kinds",
        "form",
    }
)


def _tokens_short(phrase: str, min_len: int) -> bool:
    for tok in phrase.split():
        t = _PUNCT_STRIP.sub("", tok)
        if len(t) < min_len:
            return True
    return False


def _is_generic_two_word(phrase: str) -> bool:
    parts = phrase.split()
    if len(parts) != 2:
        return False
    a, b = parts[0], parts[1]
    return a in GENERIC_ADJECTIVES and b in GENERIC_HEAD_NOUNS


def filter_phrase(
    phrase: str,
    *,
    global_counts: Counter[str],
    stopwords: set[str],
    min_token_len: int,
    min_global_freq: int,
) -> bool:
    if min_global_freq > 1 and global_counts.get(phrase, 0) < min_global_freq:
        return False
    if _tokens_short(phrase, min_token_len):
        return False
    toks = phrase.split()
    if toks[0] in stopwords or toks[-1] in stopwords:
        return False
    if toks[-1] in BAD_TAIL_TOKENS:
        return False
    # "atmospheric pressure" + "matter" (cut over three nouns) — not a term; similar chunks
    if "pressure matter" in phrase or phrase.startswith("study "):
        return False
    # Verbs often mistagged as nouns (or as NNS) — drop such tail words in a term span
    if toks[-1] in VERB_LIKE_TAIL_WORDS:
        return False
    if _is_generic_two_word(phrase):
        return False
    return True


# --- Quality filter (remove weak / generic / prefix–suffix fragments) ---------


def _is_proper_prefix_words(short: str, long: str) -> bool:
    sw, lw = short.split(), long.split()
    if len(sw) >= len(lw) or len(sw) < 2:
        return False
    return lw[: len(sw)] == sw


def _is_proper_suffix_words(short: str, long: str) -> bool:
    sw, lw = short.split(), long.split()
    if len(sw) >= len(lw) or len(sw) < 2:
        return False
    return lw[-len(sw) :] == sw


def get_never_subsume_phrases() -> set[str]:
    """Phrases that must not be removed as ‘fragments’ (e.g. gold + protected)."""
    return {preprocess_text(g) for g in GOLD_TERMS} | set(PROTECTED_TERMS)


def apply_quality_filter(
    ordered_unique: list[str],
    *,
    protected: frozenset[str] = PROTECTED_TERMS,
    never_subsume: set[str] | None = None,
) -> list[str]:
    """
    - Drop known weak exact phrases (EXACT_LOW_QUALITY) unless protected / never_subsume.
    - If a shorter phrase is a *proper* prefix or suffix of a longer *other* candidate
      in the same list, drop the shorter (incomplete head/tail fragment), unless protected.
    """
    if never_subsume is None:
        never_subsume = get_never_subsume_phrases()
    pool = set(ordered_unique)
    drop: set[str] = set()
    for c in ordered_unique:
        if c in protected or c in never_subsume:
            continue
        if c in EXACT_LOW_QUALITY:
            drop.add(c)
    for c in ordered_unique:
        if c in protected or c in never_subsume or c in drop:
            continue
        for d in pool:
            if c == d or d in drop:
                continue
            if _is_proper_prefix_words(c, d) or _is_proper_suffix_words(c, d):
                drop.add(c)
                break
    return [c for c in ordered_unique if c not in drop]


# --- Extraction + filtering ---------------------------------------------------


def extract_candidate_phrases(
    text: str,
    *,
    min_token_len: int = MIN_TOKEN_LEN,
    min_global_freq: int = MIN_GLOBAL_FREQ,
    min_w: int = MIN_SPAN_WORDS,
    max_w: int = MAX_SPAN_WORDS,
    stopwords: set[str] | None = None,
) -> tuple[
    list[list[str]],
    list[str],
    Counter[str],
    list[str],
    list[str],
    int,
]:
    """
    Returns:
        per_sentence_final: per-sentence phrase lists (after base + quality filter)
        unique_final: unique phrases, first-seen order (after both filters)
        raw_counts, all_raw_mentions, unique_raw: raw extraction stats
        n_unique_before_quality: unique count after base filter only
    """
    text = preprocess_text(text)
    stops = stopwords if stopwords is not None else get_stopwords()
    sents = tokenize_sentences(text)

    all_raw_mentions: list[str] = []
    raw_per_sent: list[list[str]] = []
    for sent in sents:
        found = extract_exact_spans_from_sentence(sent, min_w=min_w, max_w=max_w)
        raw_per_sent.append(found)
        all_raw_mentions.extend(found)

    raw_counts: Counter[str] = Counter(all_raw_mentions)
    seen_r: set[str] = set()
    unique_raw: list[str] = []
    for p in all_raw_mentions:
        if p not in seen_r:
            seen_r.add(p)
            unique_raw.append(p)

    per_sentence_filtered: list[list[str]] = []
    for sent_phrases in raw_per_sent:
        kept = [
            p
            for p in sent_phrases
            if filter_phrase(
                p,
                global_counts=raw_counts,
                stopwords=stops,
                min_token_len=min_token_len,
                min_global_freq=min_global_freq,
            )
        ]
        per_sentence_filtered.append(kept)

    seen2: set[str] = set()
    unique_before_quality: list[str] = []
    for sent in per_sentence_filtered:
        for p in sent:
            if p not in seen2:
                seen2.add(p)
                unique_before_quality.append(p)

    n_unique_before_quality = len(unique_before_quality)
    unique_final = apply_quality_filter(unique_before_quality)
    keep = set(unique_final)
    per_sentence_final = [[p for p in s if p in keep] for s in per_sentence_filtered]

    return (
        per_sentence_final,
        unique_final,
        raw_counts,
        all_raw_mentions,
        unique_raw,
        n_unique_before_quality,
    )


# --- Scoring (ranking helper used by TF-IDF and co-occurrence) ----------------

def rank_by_scores(scores: dict[str, float], descending: bool = True) -> list[str]:
    items = sorted(scores.items(), key=lambda x: (-x[1], x[0]) if descending else (x[1], x[0]))
    return [p for p, _ in items]


# --- Method 1: TF-IDF ---------------------------------------------------------


def compute_tfidf(
    per_sentence_phrases: Sequence[Sequence[str]],
) -> tuple[dict[str, float], list[str]]:
    scores = tfidf_phrase_scores(per_sentence_phrases)
    ranked = rank_by_scores(scores)
    return scores, ranked


def tfidf_phrase_scores(per_sentence_phrases: Sequence[Sequence[str]]) -> dict[str, float]:
    if not per_sentence_phrases:
        return {}

    phrase_to_idx: dict[str, int] = {}
    for phrases in per_sentence_phrases:
        for p in phrases:
            if p not in phrase_to_idx:
                phrase_to_idx[p] = len(phrase_to_idx)
    if not phrase_to_idx:
        return {}

    n_docs = len(per_sentence_phrases)
    n_terms = len(phrase_to_idx)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for doc_id, phrases in enumerate(per_sentence_phrases):
        for phrase, c in Counter(phrases).items():
            if phrase in phrase_to_idx:
                rows.append(doc_id)
                cols.append(phrase_to_idx[phrase])
                data.append(float(c))

    X = csr_matrix((data, (rows, cols)), shape=(n_docs, n_terms))
    tfidf = TfidfTransformer(sublinear_tf=False, norm="l2", use_idf=True, smooth_idf=True)
    X_tfidf = tfidf.fit_transform(X)
    col_sums = np.asarray(X_tfidf.sum(axis=0)).ravel()
    idx_to_phrase = [None] * n_terms
    for p, j in phrase_to_idx.items():
        idx_to_phrase[j] = p
    return {idx_to_phrase[j]: float(col_sums[j]) for j in range(n_terms)}


# --- Co-occurrence -------------------------------------------------------------


def build_cooccurrence(
    tokens: Sequence[str],
    window_size: int = 5,
) -> dict[tuple[str, str], int]:
    cooc: dict[tuple[str, str], int] = defaultdict(int)
    n = len(tokens)
    if n < 2:
        return dict(cooc)
    w = max(2, window_size)
    for start in range(0, max(1, n - w + 1)):
        win = tokens[start : start + w]
        L = len(win)
        for i in range(L):
            for j in range(i + 1, L):
                a, b = win[i], win[j]
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                cooc[(a, b)] += 1
    return dict(cooc)


def cooccurrence_phrase_score(phrase: str, cooc: dict[tuple[str, str], int]) -> int:
    parts = phrase.split()
    if len(parts) < 2:
        return 0
    total = 0
    for i in range(len(parts) - 1):
        a, b = parts[i], parts[i + 1]
        if a > b:
            a, b = b, a
        total += cooc.get((a, b), 0)
    return total


def compute_cooccurrence(
    text: str,
    candidates: Sequence[str],
    window_size: int = 5,
) -> tuple[dict[str, float], list[str]]:
    text = preprocess_text(text)
    tokens: list[str] = []
    for sent in tokenize_sentences(text):
        tokens.extend(word_tokenize(sent))
    cooc = build_cooccurrence(tokens, window_size=window_size)
    raw_scores = {p: float(cooccurrence_phrase_score(p, cooc)) for p in candidates}
    return raw_scores, rank_by_scores(raw_scores)


# --- Normalization (evaluation) -----------------------------------------------

# Penn -> WordNet for lemmatization
def _ptb_to_wn(t: str) -> str:
    if t.startswith("J"):
        return "a"
    if t.startswith("N"):
        return "n"
    if t.startswith("R"):
        return "r"
    if t.startswith("V"):
        return "v"
    return "n"


def strip_phrase_punct(phrase: str) -> str:
    """Lowercase, strip per-token leading/trailing punctuation, collapse space."""
    toks: list[str] = []
    for w in phrase.split():
        t = _PUNCT_STRIP.sub("", w.lower())
        if t:
            toks.append(t)
    return " ".join(toks)


def normalize_term_for_eval_raw(phrase: str) -> str:
    """
    (1) lowercase (2) strip punctuation from each token. No lemmatization.
    For strict string-style matching after cleanup.
    """
    return strip_phrase_punct(phrase).strip()


def normalize_term_for_eval_lemma(phrase: str) -> str:
    """
    Full token lemmatization (all POS). Prefer normalize_term_for_eval_noun_lemma for
    term matching; kept for older comparisons.
    """
    s = strip_phrase_punct(phrase)
    if not s:
        return ""
    lem = get_lemmatizer()
    wds = word_tokenize(s)
    if not wds:
        return ""
    tags = [t for _, t in pos_tag(wds)]
    lem_parts: list[str] = []
    for w, t in zip(wds, tags):
        wn_pos = _ptb_to_wn(t)
        if wn_pos in ("n", "a", "r", "v"):
            lem_parts.append(lem.lemmatize(w, pos=wn_pos))
        else:
            lem_parts.append(lem.lemmatize(w))
    return " ".join(lem_parts).lower()


def normalize_term_for_eval_noun_lemma(phrase: str) -> str:
    """
    Evaluation match key: lowercase, strip punct, lemmatize **nouns** to singular; other
    tokens left lowercased. Aligns 'planetary bodies' with 'planetary body', 'telescopes' with
    'telescope', without lemmatizing adjectives.
    """
    s = strip_phrase_punct(phrase)
    if not s:
        return ""
    wds = word_tokenize(s)
    if not wds:
        return ""
    lem = get_lemmatizer()
    tags = [t for _, t in pos_tag(wds)]
    out: list[str] = []
    for w, t in zip(wds, tags):
        if t in ("NN", "NNS", "NNP", "NNPS", "FW"):
            out.append(lem.lemmatize(w, pos="n").lower())
        else:
            out.append(w.lower())
    return " ".join(out)


# Back-compat alias: evaluation uses noun-lemmatized match
def normalize_term(s: str) -> str:
    return normalize_term_for_eval_noun_lemma(s)


def evaluate_precision_at_k(
    ranked_terms: Sequence[str],
    gold_terms: Iterable[str],
    ks: Sequence[int] = K_DEFAULTS,
    *,
    use_lemmatized_match: bool = True,
) -> dict[int, float]:
    if use_lemmatized_match:
        norm_pred = normalize_term_for_eval_noun_lemma
        norm_gold = normalize_term_for_eval_noun_lemma
    else:
        norm_pred = normalize_term_for_eval_raw
        norm_gold = normalize_term_for_eval_raw
    gold = {norm_gold(g) for g in gold_terms if norm_gold(g)}
    out: dict[int, float] = {}
    n = len(ranked_terms)
    for k in ks:
        if k <= 0:
            out[k] = 0.0
            continue
        hits = 0
        for i in range(min(k, n)):
            if norm_pred(ranked_terms[i]) in gold:
                hits += 1
        out[k] = hits / float(k)
    return out


def precision_at_k(
    ranked_terms: Sequence[str],
    gold_terms: Iterable[str],
    k: int,
    *,
    use_lemmatized_match: bool = True,
) -> float:
    return evaluate_precision_at_k(
        ranked_terms, gold_terms, ks=(k,), use_lemmatized_match=use_lemmatized_match
    )[k]


# --- Pipeline & reporting -----------------------------------------------------


def run_pipeline(
    text: str,
    window_size: int = 5,
    min_global_freq: int = MIN_GLOBAL_FREQ,
    ks: Sequence[int] = K_DEFAULTS,
) -> dict:
    per_sent, unique_candidates, raw_counts, all_raw, unique_raw, n_before_quality = (
        extract_candidate_phrases(text, min_global_freq=min_global_freq)
    )
    return run_pipeline_from_parts(
        per_sent,
        unique_candidates,
        raw_counts,
        all_raw,
        unique_raw,
        n_before_quality,
        text,
        window_size=window_size,
        ks=ks,
    )


def run_pipeline_from_parts(
    per_sent: list[list[str]],
    unique_candidates: list[str],
    raw_counts: Counter[str],
    all_raw_mentions: list[str],
    unique_raw: list[str],
    n_unique_before_quality: int,
    text: str,
    window_size: int = 5,
    ks: Sequence[int] = K_DEFAULTS,
) -> dict:
    tfidf_scores, tfidf_ranked = compute_tfidf(per_sent)
    cooc_scores, cooc_ranked = compute_cooccurrence(text, unique_candidates, window_size=window_size)
    top: dict[str, dict[int, list[str]]] = {"tfidf": {}, "cooccurrence": {}}
    for method, ranked in (("tfidf", tfidf_ranked), ("cooccurrence", cooc_ranked)):
        for kk in ks:
            top[method][kk] = ranked[:kk]
    return {
        "per_sentence_phrases": per_sent,
        "unique_candidates": unique_candidates,
        "n_unique_before_quality": n_unique_before_quality,
        "raw_counts": raw_counts,
        "all_raw_mentions": all_raw_mentions,
        "unique_raw": unique_raw,
        "tfidf_scores": tfidf_scores,
        "tfidf_ranked": tfidf_ranked,
        "cooccurrence_scores": cooc_scores,
        "cooccurrence_ranked": cooc_ranked,
        "top_k": top,
    }


def format_top_k_report(
    tfidf_ranked: Sequence[str],
    cooc_ranked: Sequence[str],
    ks: Sequence[int] = K_DEFAULTS,
) -> str:
    lines: list[str] = []
    for method_name, ranked in (("TF-IDF", tfidf_ranked), ("Co-occurrence", cooc_ranked)):
        for kk in ks:
            chunk = ranked[:kk]
            lines.append(
                f"{method_name} Top {kk} ({len(chunk)} shown, {len(ranked)} ranked total):"
            )
            for i, p in enumerate(chunk, 1):
                lines.append(f"  {i}. {p}")
            lines.append("")
    return "\n".join(lines).rstrip()


def format_paper_precision_table(
    title: str,
    p_t: dict[int, float],
    p_c: dict[int, float],
    ks: Sequence[int] = K_DEFAULTS,
) -> str:
    """Compact two-row table for a paper: Method and P@10 P@20 P@50 columns."""
    head = f"{'Method':<16}" + "".join(f"{'P@' + str(k):>9}" for k in ks)
    r1 = f"{'TF-IDF':<16}" + "".join(f"{p_t[k]:>9.3f}" for k in ks)
    r2 = f"{'Co-occurrence':<16}" + "".join(f"{p_c[k]:>9.3f}" for k in ks)
    body = "\n".join([head, r1, r2])
    if title:
        return title + "\n\n" + body
    return body


def format_top10_only(
    tfidf_ranked: Sequence[str],
    cooc_ranked: Sequence[str],
) -> str:
    lines = ["TF-IDF Top 10:", ""]
    for i, p in enumerate(tfidf_ranked[:10], 1):
        lines.append(f"  {i:2d}. {p}")
    lines.extend(["", "Co-occurrence Top 10:", ""])
    for i, p in enumerate(cooc_ranked[:10], 1):
        lines.append(f"  {i:2d}. {p}")
    return "\n".join(lines)


def build_final_results_text(
    n_before_quality: int,
    n_after_quality: int,
    final_candidates: Sequence[str],
    tfidf_ranked: Sequence[str],
    cooc_ranked: Sequence[str],
    gold: Iterable[str],
    ks: Sequence[int] = K_DEFAULTS,
) -> str:
    """Full narrative for `final_results.txt` (paper-ready)."""
    p_raw_t = evaluate_precision_at_k(
        tfidf_ranked, gold, ks=ks, use_lemmatized_match=False
    )
    p_raw_c = evaluate_precision_at_k(
        cooc_ranked, gold, ks=ks, use_lemmatized_match=False
    )
    p_nm_t = evaluate_precision_at_k(
        tfidf_ranked, gold, ks=ks, use_lemmatized_match=True
    )
    p_nm_c = evaluate_precision_at_k(
        cooc_ranked, gold, ks=ks, use_lemmatized_match=True
    )
    t1, m1 = error_analysis(tfidf_ranked, gold, k=10, use_lemmatized_match=True)
    t2, m2 = error_analysis(cooc_ranked, gold, k=10, use_lemmatized_match=True)
    bl: list[str] = [
        "Terminology extraction — final results",
        "(Original = lowercase + strip punctuation. Normalized = noun lemmatization for singular/plural.)",
        "",
        "1. Candidate counts",
        f"   Unique after base filter:    {n_before_quality}",
        f"   Unique after quality filter: {n_after_quality}",
        "",
        "2. Precision@k (original / surface match)",
    ]
    for line in format_paper_precision_table("", p_raw_t, p_raw_c, ks).split("\n"):
        bl.append("   " + line)
    bl.extend(
        [
            "",
            "3. Precision@k (normalized: noun lemmatization, gold and predictions)",
        ]
    )
    for line in format_paper_precision_table("", p_nm_t, p_nm_c, ks).split("\n"):
        bl.append("   " + line)
    bl.extend(["", "4. Top-10 terms", "", format_top10_only(tfidf_ranked, cooc_ranked)])
    bl.extend(
        [
            "",
            "5. Error analysis (top-10 vs gold, normalized noun match)",
            "",
            "   TF — incorrect (predicted in top-10, not in gold), examples:",
        ]
    )
    for x in t1:
        bl.append(f"      - {x}")
    bl.append("   TF — missed gold (not in top-10), examples:")
    for x in m1:
        bl.append(f"      - {x}")
    bl.extend(
        [
            "",
            "   Co-occurrence — incorrect (examples):",
        ]
    )
    for x in t2:
        bl.append(f"      - {x}")
    bl.append("   Co-occurrence — missed gold (examples):")
    for x in m2:
        bl.append(f"      - {x}")
    bl.extend(
        [
            "",
            f"6. All final candidate phrases (n={len(final_candidates)})",
        ]
    )
    for i, p in enumerate(final_candidates, 1):
        bl.append(f"   {i:3d}. {p}")
    return "\n".join(bl)


def format_dual_evaluation_table(
    tfidf_ranked: Sequence[str],
    cooc_ranked: Sequence[str],
    gold: Iterable[str],
    ks: Sequence[int] = K_DEFAULTS,
) -> str:
    """P@k: raw string vs noun-lemmatized (singular/plural alignment)."""
    pr_t = evaluate_precision_at_k(
        tfidf_ranked, gold, ks=ks, use_lemmatized_match=False
    )
    pl_t = evaluate_precision_at_k(
        tfidf_ranked, gold, ks=ks, use_lemmatized_match=True
    )
    pr_c = evaluate_precision_at_k(
        cooc_ranked, gold, ks=ks, use_lemmatized_match=False
    )
    pl_c = evaluate_precision_at_k(
        cooc_ranked, gold, ks=ks, use_lemmatized_match=True
    )
    h1 = (
        ["Method", "Match"]
        + [f"P@{k}" for k in ks]
    )
    rows = [
        ["TF-IDF", "raw (lowercase + punct)"]
        + [f"{pr_t[k]:.3f}" for k in ks],
        ["TF-IDF", "norm (noun lemma)"]
        + [f"{pl_t[k]:.3f}" for k in ks],
        ["Co-occurrence", "raw (lowercase + punct)"]
        + [f"{pr_c[k]:.3f}" for k in ks],
        ["Co-occurrence", "norm (noun lemma)"]
        + [f"{pl_c[k]:.3f}" for k in ks],
    ]
    w = [len(h1[i]) for i in range(len(h1))]
    for r in rows:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(c))
    out = ["  ".join(h1[i].ljust(w[i]) for i in range(len(h1)))]
    for r in rows:
        out.append("  ".join(r[i].ljust(w[i]) for i in range(len(r))))
    return "\n".join(out)


def error_analysis(
    ranked: Sequence[str],
    gold_terms: Iterable[str],
    k: int = 10,
    *,
    use_lemmatized_match: bool = True,
) -> tuple[list[str], list[str]]:
    """
    (incorrect top-k predictions (surface), missed gold (one surface string per gold norm)).
    """
    if use_lemmatized_match:
        norm = normalize_term_for_eval_noun_lemma
    else:
        norm = normalize_term_for_eval_raw
    gmap: dict[str, str] = {}
    for g in gold_terms:
        ng = norm(g)
        if ng and ng not in gmap:
            gmap[ng] = g.strip()
    gset = set(gmap)
    pred_not_in_gold: list[str] = []
    for i in range(min(k, len(ranked))):
        r = ranked[i]
        if norm(r) not in gset:
            pred_not_in_gold.append(r)
    topk_norms = {norm(ranked[i]) for i in range(min(k, len(ranked)))}
    gold_missed = [gmap[gn] for gn in sorted(gset) if gn not in topk_norms]
    return pred_not_in_gold, gold_missed


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_result_files(
    out_dir: Path,
    tfidf_ranked: Sequence[str],
    cooc_ranked: Sequence[str],
    gold: Iterable[str],
    ks: Sequence[int] = K_DEFAULTS,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_text(out_dir / "results_tfidf.txt", "\n".join(f"{i+1}\t{p}" for i, p in enumerate(tfidf_ranked)) + "\n")
    save_text(out_dir / "results_cooccurrence.txt", "\n".join(f"{i+1}\t{p}" for i, p in enumerate(cooc_ranked)) + "\n")

    report_lines: list[str] = [
        "Terminology extraction — results",
        "",
        f"TF-IDF ranked: {len(tfidf_ranked)} | Co-occurrence ranked: {len(cooc_ranked)}",
        "",
        format_top_k_report(tfidf_ranked, cooc_ranked, ks=ks),
        "",
        "=== Evaluation: raw vs normalized (lemmatize) ===",
        format_dual_evaluation_table(tfidf_ranked, cooc_ranked, gold, ks=ks),
        "",
        "=== Example errors @10 (lemma-normalized) ===",
    ]
    p1, g1 = error_analysis(tfidf_ranked, gold, k=10, use_lemmatized_match=True)
    p2, g2 = error_analysis(cooc_ranked, gold, k=10, use_lemmatized_match=True)
    report_lines.append("TF-IDF — top-10 predicted not in gold (normalized):")
    report_lines.extend(f"  - {x}" for x in p1)
    report_lines.append("TF-IDF — gold missing from top-10:")
    report_lines.extend(f"  - {x}" for x in g1)
    report_lines.append("")
    report_lines.append("Co-occurrence — top-10 predicted not in gold (normalized):")
    report_lines.extend(f"  - {x}" for x in p2)
    report_lines.append("Co-occurrence — gold missing from top-10:")
    report_lines.extend(f"  - {x}" for x in g2)
    report_lines.append("")
    save_text(out_dir / "results.txt", "\n".join(report_lines))


# Max lines of raw candidates printed to stdout (full list always in candidates_all_raw.txt)
MAX_PRINT_RAW_LINES = 500


EXAMPLE_PARAGRAPH = """
The solar system consists of the Sun and the objects that orbit it, including
planetary bodies such as Earth and Mars. A planetary body may be small. Impact craters
and mean density are discussed. The terrestrial planets differ from the giant planets.
Terrestrial-like bodies are interesting. The surface temperature and high atmospheric
pressure affect the greenhouse effect. The rotation period and orbital period
are measured. Synchronous rotation is common. Surface minerals on the near side
and far side vary. A planetary formation process takes long. Space telescopes
observe gravitational waves and black holes. The interstellar medium is cold.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Astrophysics terminology extraction (TF-IDF + co-occurrence).")
    parser.add_argument(
        "corpus",
        nargs="?",
        default=None,
        help=f"Path to plain-text corpus (default: {CORPUS_FILE} next to this script, if present; else short demo).",
    )
    args = parser.parse_args()
    ensure_nltk_data()
    script_dir = Path(__file__).resolve().parent
    if args.corpus:
        corpus_path = Path(args.corpus)
    else:
        default_p = script_dir / CORPUS_FILE
        corpus_path = default_p if default_p.is_file() else None

    if corpus_path and corpus_path.is_file():
        text = load_corpus_text(corpus_path)
        print(f"Loaded corpus: {corpus_path} ({len(text)} characters)\n")
    else:
        text = EXAMPLE_PARAGRAPH.strip()
        if corpus_path:
            print(f"Note: file not found: {corpus_path} — using built-in example paragraph.\n")
        else:
            print(f"Note: no {CORPUS_FILE} found — using built-in example paragraph. Add {script_dir / CORPUS_FILE} for your pages.\n")

    out = run_pipeline(text)
    raw_counts: Counter[str] = out["raw_counts"]
    unique_raw: list[str] = out["unique_raw"]
    all_raw: list[str] = out["all_raw_mentions"]
    unique_f = out["unique_candidates"]
    n_before_q = int(out["n_unique_before_quality"])

    print("=== Candidate statistics ===")
    print("Total raw exact-span mentions (before filter):", len(all_raw))
    print("Number of unique raw candidates:", len(unique_raw))
    print("Unique after base (POS/stop/verb) filter:", n_before_q)
    print("Unique after quality filter:             ", len(unique_f))
    print()

    print("All raw candidate phrase mentions (before filter), corpus order — full list in candidates_all_raw.txt")
    if len(all_raw) <= MAX_PRINT_RAW_LINES:
        for p in all_raw:
            print(p)
    else:
        print(f"  (Console suppressed: {len(all_raw)} lines; open candidates_all_raw.txt.)")
    print()

    print("First 50 unique raw candidates (first-seen order):")
    for p in unique_raw[:50]:
        print(f"  {p}")
    if len(unique_raw) > 50:
        print(f"  ... and {len(unique_raw) - 50} more")
    print()

    print("First 50 unique filtered candidates (after quality pass):")
    for p in unique_f[:50]:
        print(f"  {p}")
    if len(unique_f) > 50:
        print(f"  ... and {len(unique_f) - 50} more")
    print()
    print("All final candidate phrases (after quality filter):")
    for i, p in enumerate(unique_f, 1):
        print(f"  {i:3d}. {p}")
    print()

    tfidf_ranked = out["tfidf_ranked"]
    cooc_ranked = out["cooccurrence_ranked"]
    gold = GOLD_TERMS

    p_raw_t = evaluate_precision_at_k(tfidf_ranked, gold, ks=K_DEFAULTS, use_lemmatized_match=False)
    p_raw_c = evaluate_precision_at_k(cooc_ranked, gold, ks=K_DEFAULTS, use_lemmatized_match=False)
    p_nm_t = evaluate_precision_at_k(tfidf_ranked, gold, ks=K_DEFAULTS, use_lemmatized_match=True)
    p_nm_c = evaluate_precision_at_k(cooc_ranked, gold, ks=K_DEFAULTS, use_lemmatized_match=True)

    print("=== Evaluation (P@k) — original (surface) ===")
    print(format_paper_precision_table("", p_raw_t, p_raw_c, K_DEFAULTS))
    print()
    print("=== Evaluation (P@k) — normalized (noun lemmatization) ===")
    print(format_paper_precision_table("", p_nm_t, p_nm_c, K_DEFAULTS))
    print()
    print(format_top10_only(tfidf_ranked, cooc_ranked))
    print()
    print(format_top_k_report(tfidf_ranked, cooc_ranked))
    print("Ranked list lengths — TF-IDF:", len(tfidf_ranked), " Co-occurrence:", len(cooc_ranked))
    print()

    print("=== Full evaluation table (4 rows) ===")
    print(format_dual_evaluation_table(tfidf_ranked, cooc_ranked, gold))
    print()

    p1, g1 = error_analysis(tfidf_ranked, gold, k=10, use_lemmatized_match=True)
    print("Error analysis @10 (noun-lemma match) — TF-IDF")
    print("  Incorrect (in top-10, not in gold), examples:", p1)
    print("  Missed gold (not in top-10), examples:          ", g1)
    p2, g2 = error_analysis(cooc_ranked, gold, k=10, use_lemmatized_match=True)
    print("Error analysis @10 — Co-occurrence")
    print("  Incorrect, examples: ", p2)
    print("  Missed gold, examples:", g2)
    print()

    write_result_files(script_dir, tfidf_ranked, cooc_ranked, gold)
    save_text(script_dir / "candidates_all_raw.txt", "\n".join(all_raw) + "\n")
    final_txt = build_final_results_text(
        n_before_q, len(unique_f), unique_f, tfidf_ranked, cooc_ranked, gold, ks=K_DEFAULTS
    )
    save_text(script_dir / "final_results.txt", final_txt)
    print(
        f"Wrote {script_dir / 'candidates_all_raw.txt'}, {script_dir / 'results_tfidf.txt'}, "
        f"{script_dir / 'results_cooccurrence.txt'}, {script_dir / 'results.txt'}, "
        f"{script_dir / 'final_results.txt'}"
    )


if __name__ == "__main__":
    main()
