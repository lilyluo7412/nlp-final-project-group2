from __future__ import annotations

import re
from pathlib import Path

import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

_WS_RE = re.compile(r"\s+")
_PUNCT_STRIP = re.compile(r"^[^a-z0-9]+|[^a-z0-9]+$", re.IGNORECASE)


def ensure_nltk_data() -> None:
    for resource, download_name in (
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
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


def normalize_text(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return _WS_RE.sub(" ", text.strip())

def normalize_term(term: str) -> str:
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    toks: list[str] = []
    for t in term.split():
        s = _PUNCT_STRIP.sub("", t.lower())
        if s:
            toks.append(lem.lemmatize(s, pos="n"))
    return " ".join(toks)


def tokenize_and_tag(text: str) -> list[list[tuple[str, str]]]:
    ensure_nltk_data()
    out: list[list[tuple[str, str]]] = []
    for sent in sent_tokenize(text):
    # skip timestamp lines and very short header sentences
        if re.search(r'\[\d{2}:\d{2}:\d{2}\]', sent):
            continue
        tokens = word_tokenize(sent)
        if not tokens:
            continue
        out.append(pos_tag(tokens))
    return out


def load_documents(raw_root: Path) -> dict[str, str]:
    """
    Load all raw .txt under data/raw layout:
      - data/raw/dev/*.txt
      - data/raw/test/*.txt
    If neither subdir has files, fall back to raw_root/*.txt (flat).
    """
    docs: dict[str, str] = {}
    dev_dir = raw_root / "dev"
    test_dir = raw_root / "test"
    for sub in (dev_dir, test_dir):
        if sub.is_dir():
            for p in sorted(sub.glob("*.txt")):
                docs[p.stem] = normalize_text(p.read_text(encoding="utf-8").lstrip("\ufeff"))
    if docs:
        return docs
    for p in sorted(raw_root.glob("*.txt")):
        docs[p.stem] = normalize_text(p.read_text(encoding="utf-8").lstrip("\ufeff"))
    return docs
