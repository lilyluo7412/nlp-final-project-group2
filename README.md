# Terminology extraction — NLP final project

Course **final project**: multi-word **noun-phrase** terminology extraction from **English astrophysics textbook** text, comparing two baselines: **TF-IDF** (one sentence = one document) and **sliding-window co-occurrence**.

## Textbook (domain source)

The analysis targets material from:

**Rothery, D. A., McBride, N., & Gilmour, I.** (eds.). *An Introduction to the Solar System* (3rd ed.). **Cambridge University Press**, **2018**. **ISBN-13: 978-1-108-43084-5** (9781108430845).

*(Publisher metadata may also list additional editors or contributors on the title page; the project uses plain-text extracts in `corpus.txt` from selected pages of this edition—not the PDF stored in the repo.)*

## Team (Group 2)

| Name | GitHub |
|------|--------|
| Lily Luo | [lilyluo7412](https://github.com/lilyluo7412) |
| Nandana Krishnaraj | [nandanakrish9](https://github.com/nandanakrish9) |
| Nicole Jin | [nicolejin099](https://github.com/nicolejin099) |
| Bella Diallo | [hbd9577](https://github.com/hbd9577) |

## Quick start

```bash
cd nlp-final-project-group2
python3 -m pip install -r requirements.txt
```

Put your textbook text in **`corpus.txt`** (UTF-8, multiple pages allowed) in this directory, then:

```bash
python3 terminology_extraction.py
```

Or pass a file path:

```bash
python3 terminology_extraction.py /path/to/your_corpus.txt
```

If `corpus.txt` is missing, the script falls back to a short built-in example paragraph.

## Output files (same folder after a run)

| File | Description |
|------|-------------|
| **`final_results.txt`** | Paper-ready: candidate counts, P@10/20/50 (raw vs noun-normalized), Top-10, short error analysis, final candidate list |
| `results_tfidf.txt` / `results_cooccurrence.txt` | Full ranked output for each method |
| `candidates_all_raw.txt` | All raw candidates before some filters (debugging) |

These generated `.txt` outputs are **gitignored** (see `.gitignore`); clone the repo, add `corpus.txt`, and run the script to recreate them locally (or copy into your report as needed).

## What teammates can edit

- **`GOLD_TERMS`** (top of `terminology_extraction.py`): hand-annotated terms for **Precision@k**; includes e.g. `near side` / `far side`.
- **`corpus.txt`**: replace with plain text from the pages you label, then re-run.
- **`EXACT_LOW_QUALITY`**: blocklist for overly generic strings (e.g. `planetary formation process`) without changing the main pipeline.

## Method overview (no code changes required)

1. **Candidates**: consecutive 2–4 tokens per sentence + POS patterns (e.g. compound nouns, `(Mod)*(Noun)+`), then a base filter and a **quality** filter.
2. **Evaluation**: match against `GOLD_TERMS`; the **normalized** line lemmatizes **nouns** to reduce singular/plural mismatches (e.g. bodies/body).

## Dependencies

Python 3. See `requirements.txt` (`nltk`, `scikit-learn`, `numpy`, `scipy`). The first run downloads NLTK data (tokenization, POS, stopwords, WordNet) as needed.

---

*NLP final project — Group 2 — astrophysics textbook terminology extraction*
