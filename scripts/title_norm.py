#!/usr/bin/env python3
"""Title normalization: strip code prefix, extract disambiguator, lemmatize.

Public API:
  normalize_title(title: str) -> {stem, disambig, normalized_full}
  has_disambig_conflict(a: dict, b: dict) -> bool

pymorphy3 is optional; falls back to lowercased tokens if unavailable.
"""
from __future__ import annotations

import re
from functools import lru_cache

try:
    import pymorphy3

    _morph = pymorphy3.MorphAnalyzer()
except Exception:  # pragma: no cover
    _morph = None


# TMS-style id prefix at the very start, e.g. "ППП-Т1213 ", "ABC-T42 ".
CODE_PREFIX_RE = re.compile(r"^[A-Za-zА-Яа-яЁё]{2,10}-[A-Za-zА-Яа-яЁё]*\d+\s*")

# Disambiguator patterns — first match wins, checked at the TAIL of the title.
#   (по счету), — по карте, : для физлица
_DISAMBIG_PATTERNS = (
    re.compile(r"\(([^()]+)\)\s*$"),
    re.compile(r"\s+[—–-]\s+([^—–-]+)$"),
    re.compile(r":\s+([^:]+)$"),
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@lru_cache(maxsize=20000)
def _lemma(token: str) -> str:
    if _morph is None:
        return token
    try:
        return _morph.parse(token)[0].normal_form
    except Exception:
        return token


def _normalize_text(text: str) -> str:
    """Lowercase + lemmatize each word, drop non-word chars."""
    return " ".join(_lemma(t.lower()) for t in _WORD_RE.findall(text))


def normalize_title(title: str) -> dict:
    """Return dict(stem, disambig, normalized_full). All values lemmatized + lowercased."""
    t = CODE_PREFIX_RE.sub("", title or "").strip()
    disambig_raw = ""
    stem_raw = t
    for pat in _DISAMBIG_PATTERNS:
        m = pat.search(t)
        if m:
            disambig_raw = m.group(1).strip()
            stem_raw = t[: m.start()].strip()
            break

    stem = _normalize_text(stem_raw)
    disambig = _normalize_text(disambig_raw) if disambig_raw else ""
    normalized_full = f"{stem} | {disambig}" if disambig else stem
    return {"stem": stem, "disambig": disambig, "normalized_full": normalized_full}


def has_disambig_conflict(a: dict, b: dict) -> bool:
    """Same stem, both disambigs non-empty and differ → disambig conflict.

    This is the strong signal for "titles look the same but distinguish by
    parenthetical variant" — e.g. "Проверка X (по счету)" vs "(по карте)".
    """
    if not a["stem"] or a["stem"] != b["stem"]:
        return False
    return bool(a["disambig"]) and bool(b["disambig"]) and a["disambig"] != b["disambig"]


if __name__ == "__main__":
    import json
    import sys

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        print(json.dumps(normalize_title(line), ensure_ascii=False))
