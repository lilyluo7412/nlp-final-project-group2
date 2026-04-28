# Automatic Terminology Extraction — Course Project (Group 2)

We extract domain multi-word noun phrases (**2–4 tokens**, Penn-style POS heuristic `(JJ)*(NN)+`), rank candidates with **TF-IDF** and a **simple window co-occurrence** baseline, and report **Precision@10 / @20 / @50** against manual gold phrases (normalized exact match).

## Team

| Name | GitHub |
|------|--------|
| Lily Luo | [lilyluo7412](https://github.com/lilyluo7412) |
| Nandana Krishnaraj | [nandanakrish9](https://github.com/nandanakrish9) |
| Nicole Jin | [nicolejin099](https://github.com/nicolejin099) |
| Bella Diallo | [hbd9577](https://github.com/hbd9577) |

---

## Repository layout

| Path | Purpose |
|------|---------|
| `data/raw/dev/`, `data/raw/test/` | One `.txt` per document UTF-8; filename stem **=** doc id |
| `data/gold_standard/gold_standard_*.tsv` | Source-of-truth tabular gold (columns: doc id + term phrase) |
| `data/splits/` | `dev_ids.txt` / `test_ids.txt` (generated from folder listings) |
| `data/annotations/` | `gold_terms_*.jsonl` (generated JSONL consumed by metrics) |
| `config.yaml` | Tunable parameters (`min_freq`, `window_size`, span length caps) |
| `src/` | Preprocessing, candidate extraction, TF-IDF, co-occurrence, evaluation |
| `scripts/` | Data wiring (`build_wired_data.py`), error analysis (`error_analysis.py`), optional validation helpers below |
| `outputs/` | Dev and test CSV metrics and rankings (regenerated each run; see archiving below) |

---

## Experiment protocol

- **Development set:** used to adjust `config.yaml` knobs (typically `min_freq`, `window_size`), inspect rankings, and analyze errors (`outputs/dev/`).
- **Test set:** used **only for final headline numbers** in `outputs/test/`; do **not** tune hyperparameters to maximize test precision (prevents leakage from held-out annotations).

Both ranking methods consume the **same** candidate extractor and identical gold normalization for fair comparison.

---

## Reproduce a run end-to-end

```bash
cd nlp-final-project-group2
python3 -m pip install -r requirements.txt
python3 scripts/build_wired_data.py       # regenerate JSONL splits from TSVs if editors changed inputs
python3 terminology_extraction.py       # unified entrypoint; writes outputs/dev + outputs/test
```

All results can be reproduced with the commands above using the provided data and configuration.

`terminology_extraction.py` invokes `src.run_pipeline`; default raw root is `data/raw/` (loads `dev/` + `test/` recursively).

Optional: **`python3 terminology_extraction.py --min-freq 1`** temporarily overrides `min_freq` in `config.yaml` (for dev-side ablations) without editing the file on disk.

---

## Configuration (`config.yaml`)

| Field | Meaning |
|-------|---------|
| `min_freq` | Per-split minimum global frequency threshold for phrases |
| `window_size` | Token window span for simple co-occurrence scoring |
| `min_len` / `max_len` | Phrase lengths (frozen at **2–4 tokens** here) |

Only adjust these on dev; finalize before locking test-stage reporting.

**Reported setting:** Results in the final report correspond to **`min_freq: 2`** (and `--min-freq` unset), which keeps the candidate pool **manageable**.

**Brief dev-only check:** `--min-freq 1` can be tried for curiosity; empirically it **slightly** raised TF‑IDF P@20 on dev but **scaled unique candidates by ~×4**, with **no change on test** Precision. Because of that noise/signal trade-off, **`min_freq=2` is the default retained for submissions**.

---

## Generated outputs (`outputs/`)

| Artifact | Contents |
|---------|----------|
| `*_ranked_terms.csv` | Rank-indexed phrases with scores (`tfidf_ranked_terms.csv`, `cooc_ranked_terms.csv`) |
| `precision_at_k_dev.csv` | Dev P@10, P@20, P@50 for TF-IDF vs co-occurrence |
| `precision_at_k_test.csv` | Test P@10, P@20, P@50 for both methods |
| `tuning_log.json` (dev) | Copies config + dev Precision metrics after a run |

---

## Archiving pinned baselines for the final report or slides

To snapshot the current CSVs:

```bash
mkdir -p outputs/archive/run_<DATE>_baseline/dev outputs/archive/run_<DATE>_baseline/test
cp outputs/dev/*.csv outputs/dev/tuning_log.json outputs/archive/run_<DATE>_baseline/dev/
cp outputs/test/*.csv outputs/archive/run_<DATE>_baseline/test/
```

An example pinned folder is **`outputs/archive/run_2026-04-28_baseline/`** (example official run from the same pipeline version).

---

## Error analysis (discussion section support)

```bash
python3 scripts/error_analysis.py --split dev  --k 15
python3 scripts/error_analysis.py --split test --k 15
```

Prints top-k false positives (predicted, not in gold union) and missed gold terms (same normalized matching as metrics).

---

## Optional Scripts

These scripts are **optional utilities** for data validation and debugging. **They are not required** to run the main pipeline. Run **from the repository root** (`cd` into this project first).

| Script | Purpose |
|--------|---------|
| `scripts/check_dev.py` | Checks that gold dev phrases appear as substrings in the matching raw dev `.txt`. |
| `scripts/check_test.py` | Same check for **test** TSV ↔ `data/raw/test/`. |
| `scripts/verify_gold_standard.py` | Runs substring checks for both dev and test gold TSV files. |

```bash
python3 scripts/check_dev.py
python3 scripts/check_test.py
python3 scripts/verify_gold_standard.py
```

---

## Notes on Annotation Consistency

Some gold terms may not appear as exact substrings in their assigned documents. This is typically due to minor formatting or wording differences (e.g., hyphenation or tokenization).

These cases are limited and are treated as minor annotation noise. They do not affect the main evaluation or system conclusions.

Substring checks in **Optional Scripts** are for validation only and **do not change** the evaluation pipeline.

---

## Presentation code highlights (what to show, not line-by-line)

| Component | One-line role |
|-----------|----------------|
| `terminology_extraction.py` | Single entry point to the full pipeline |
| `scripts/build_wired_data.py` | Converts TSV gold to JSONL, writes split id lists, filters to 2–4 token terms |
| `src/candidate_extraction.py` | Extracts 2–4 token noun-phrase candidates by POS |
| `src/scoring_tfidf.py` | TF-IDF baseline ranking over candidate phrase × document matrix |
| `src/scoring_cooccurrence.py` | Alternative ranking via simple window co-occurrence on candidates |
| `src/evaluation.py` | Precision@10/20/50 vs merged gold (normalized) |
| `scripts/error_analysis.py` | Surfaces false positives / missed gold for qualitative discussion |

---

## What to mention in the presentation

- Reproducible path from **raw documents + TSV gold** to **ranked lists + Precision@k CSVs**.
- Two comparable systems: **TF-IDF** vs **window co-occurrence** on the same candidates and gold.
- Primary metric: **Precision@10/20/50**; **dev** for exploration, **test** for final comparison only.
- Empirical pattern in this corpus: **TF-IDF generally ranks domain-relevant phrases higher**; co-occurrence often favors **high-frequency lecture / transcript boilerplate** (e.g. meta-discourse phrases), which hurts domain precision.

---

## What not to over-explain

- Do **not** walk through every function line by line.
- Avoid deep implementation digressions unless they directly support **reproducibility** or **evaluation validity**.
- Do **not** claim state-of-the-art or external benchmark leadership.
- Treat **demo-scale** or **preliminary** runs differently from **full annotated** runs in any narrative.

---

## Dependencies

Python 3.10+ recommended. Install from `requirements.txt` (`nltk`, `numpy`, `scipy`, `scikit-learn`). First execution may download NLTK tokenizer and POS tagger resources.
