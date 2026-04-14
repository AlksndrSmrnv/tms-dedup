#!/usr/bin/env python3
"""Two-signal candidate clustering: title and steps similarities evaluated separately.

Rationale: when the title is the main discriminator (e.g. identical steps but
"(по счету)" vs "(по карте)"), we must NOT treat the pair as a duplicate even
if steps match 100%. So:

  1. Normalize titles (strip code prefix, extract disambiguator).
  2. Compute sim_title (char-ngram TF-IDF over normalized title) and
     sim_steps (char-ngram TF-IDF over steps text) SEPARATELY.
  3. If titles share the same stem but different disambiguators → force
     sim_title = 0.2 (disambig-conflict override). This is the hard rule
     for the copy-paste case.
  4. Label each candidate pair using a rule table:
       duplicate_full / duplicate_partial / same_intent_diff_flow /
       suspicious_copypaste / noise
  5. Union-find per label to build groups.

Candidates come from the union of two MinHash-LSH indexes (titles + steps),
so we don't miss pairs that agree on only one signal.

Output schema (clusters.json):
  [
    {
      "group_id": "g0",
      "label": "duplicate_full" | ...,
      "test_ids": ["T1", "T2", ...],
      "pair_metrics": [
        {"a": "T1", "b": "T2",
         "sim_title": 0.92, "sim_steps": 0.88,
         "disambig_conflict": false, "label": "duplicate_full"}
      ]
    }
  ]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from title_norm import normalize_title, has_disambig_conflict  # noqa: E402


WORD_RE = re.compile(r"\w+", re.UNICODE)

# Similarity thresholds. Order matters — evaluated top-down, first match wins.
LABEL_RULES = [
    # (label, min_sim_title, min_sim_steps, max_sim_title, max_sim_steps)
    ("duplicate_full",         0.85, 0.70, 1.01, 1.01),
    ("suspicious_copypaste",   0.00, 0.85, 0.60, 1.01),  # HIGH steps, LOW title
    ("same_intent_diff_flow",  0.85, 0.00, 1.01, 0.30),  # HIGH title, LOW steps
    ("duplicate_partial",      0.70, 0.50, 1.01, 1.01),
]
NOISE = "noise"

# Disambig-conflict forces sim_title to this value regardless of raw cosine.
# Low enough to never pass the 0.70 partial threshold, but non-zero for clarity.
DISAMBIG_SIM_OVERRIDE = 0.20

# Labels that are "real duplicate candidates" (fed to LLM as such).
DUPLICATE_LABELS = {"duplicate_full", "duplicate_partial", "same_intent_diff_flow"}


def steps_text(test: dict) -> str:
    parts: list[str] = []
    for step in test.get("steps", []):
        if step.get("action"):
            parts.append(step["action"])
        if step.get("expected"):
            parts.append(step["expected"])
    return "\n".join(parts).lower()


def shingles(text: str, n: int = 3) -> set[str]:
    tokens = WORD_RE.findall(text)
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def build_minhash(text: str, num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for s in shingles(text):
        mh.update(s.encode("utf-8"))
    return mh


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def label_pair(sim_title: float, sim_steps: float) -> str:
    for label, min_t, min_s, max_t, max_s in LABEL_RULES:
        if min_t <= sim_title < max_t and min_s <= sim_steps < max_s:
            return label
    return NOISE


def _tfidf_matrix(texts: list[str]):
    # Empty strings blow up TfidfVectorizer — substitute a space.
    safe = [t if t.strip() else " " for t in texts]
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_df=0.95, sublinear_tf=True
    )
    return vec.fit_transform(safe)


def _build_lsh(texts: list[str], num_perm: int, threshold: float):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mhs: list[MinHash] = []
    for i, t in enumerate(texts):
        mh = build_minhash(t, num_perm)
        lsh.insert(str(i), mh)
        mhs.append(mh)
    return lsh, mhs


def cluster(
    tests: list[dict],
    min_size: int,
    max_size: int,
    lsh_title_threshold: float,
    lsh_steps_threshold: float,
    num_perm: int,
) -> list[dict]:
    n = len(tests)
    # Normalize titles once.
    title_norms = [normalize_title(t.get("title", "")) for t in tests]
    title_texts = [tn["normalized_full"] for tn in title_norms]
    step_texts = [steps_text(t) for t in tests]

    # Two MinHash-LSH indexes — candidates are the union of both query sets.
    lsh_t, mh_t = _build_lsh(title_texts, num_perm, lsh_title_threshold)
    lsh_s, mh_s = _build_lsh(step_texts, num_perm, lsh_steps_threshold)

    # Two TF-IDF matrices for precise similarity on candidates.
    mat_t = _tfidf_matrix(title_texts)
    mat_s = _tfidf_matrix(step_texts)

    # Collect candidate pair set (i < j) from union of LSH neighborhoods.
    candidates: set[tuple[int, int]] = set()
    for i in range(n):
        for cand in lsh_t.query(mh_t[i]):
            j = int(cand)
            if j != i:
                candidates.add((i, j) if i < j else (j, i))
        for cand in lsh_s.query(mh_s[i]):
            j = int(cand)
            if j != i:
                candidates.add((i, j) if i < j else (j, i))

    # Score pairs and assign label.
    uf_by_label: dict[str, UnionFind] = {lbl: UnionFind(n) for lbl in DUPLICATE_LABELS | {"suspicious_copypaste"}}
    pairs_by_label: dict[str, list[dict]] = {lbl: [] for lbl in uf_by_label}

    for a, b in candidates:
        raw_sim_t = float(cosine_similarity(mat_t[a], mat_t[b])[0, 0])
        sim_s = float(cosine_similarity(mat_s[a], mat_s[b])[0, 0])

        conflict = has_disambig_conflict(title_norms[a], title_norms[b])
        sim_t = DISAMBIG_SIM_OVERRIDE if conflict else raw_sim_t

        lbl = label_pair(sim_t, sim_s)
        if lbl == NOISE:
            continue

        pairs_by_label[lbl].append(
            {
                "a_idx": a,
                "b_idx": b,
                "sim_title": round(sim_t, 3),
                "sim_title_raw": round(raw_sim_t, 3),
                "sim_steps": round(sim_s, 3),
                "disambig_conflict": conflict,
            }
        )
        uf_by_label[lbl].union(a, b)

    # Build groups per label.
    out: list[dict] = []
    gid = 0
    for lbl, uf in uf_by_label.items():
        # Collect members per root, but only those touched by at least one pair of this label.
        touched: set[int] = set()
        for p in pairs_by_label[lbl]:
            touched.add(p["a_idx"])
            touched.add(p["b_idx"])
        groups: dict[int, list[int]] = {}
        for i in touched:
            groups.setdefault(uf.find(i), []).append(i)

        for members in groups.values():
            if len(members) < min_size:
                continue
            chunks = [members] if len(members) <= max_size else [
                members[k : k + max_size] for k in range(0, len(members), max_size)
            ]
            for chunk in chunks:
                if len(chunk) < min_size:
                    continue
                chunk_set = set(chunk)
                pair_metrics = [
                    {
                        "a": tests[p["a_idx"]]["id"],
                        "b": tests[p["b_idx"]]["id"],
                        "sim_title": p["sim_title"],
                        "sim_steps": p["sim_steps"],
                        "disambig_conflict": p["disambig_conflict"],
                        "label": lbl,
                    }
                    for p in pairs_by_label[lbl]
                    if p["a_idx"] in chunk_set and p["b_idx"] in chunk_set
                ]
                out.append(
                    {
                        "group_id": f"g{gid}",
                        "label": lbl,
                        "test_ids": [tests[i]["id"] for i in chunk],
                        "pair_metrics": pair_metrics,
                    }
                )
                gid += 1

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tests", type=Path)
    ap.add_argument("--out", type=Path, default=Path("clusters.json"))
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=20)
    ap.add_argument("--lsh-title-threshold", type=float, default=0.5)
    ap.add_argument("--lsh-steps-threshold", type=float, default=0.5)
    ap.add_argument("--num-perm", type=int, default=128)
    args = ap.parse_args()

    tests = json.loads(args.tests.read_text(encoding="utf-8"))
    clusters = cluster(
        tests,
        args.min_size,
        args.max_size,
        args.lsh_title_threshold,
        args.lsh_steps_threshold,
        args.num_perm,
    )
    args.out.write_text(json.dumps(clusters, ensure_ascii=False, indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for c in clusters:
        counts[c["label"]] = counts.get(c["label"], 0) + 1
    total_tests = sum(len(c["test_ids"]) for c in clusters)
    mean_size = float(np.mean([len(c["test_ids"]) for c in clusters])) if clusters else 0.0
    print(f"{len(clusters)} clusters, {total_tests} tests in clusters (mean size {mean_size:.1f}) → {args.out}")
    for lbl, cnt in sorted(counts.items()):
        print(f"  {lbl}: {cnt}")


if __name__ == "__main__":
    main()
