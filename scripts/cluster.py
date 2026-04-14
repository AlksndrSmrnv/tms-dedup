#!/usr/bin/env python3
"""Candidate-cluster detection via MinHash-LSH + TF-IDF cosine filter.

Two-stage: cheap MinHash finds near-duplicate pairs, then TF-IDF cosine
confirms similarity above --threshold. Candidates are linked via union-find
into clusters of size 2..max-size.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from embedders import get_embedder  # noqa: E402


WORD_RE = re.compile(r"\w+", re.UNICODE)


def test_text(test: dict) -> str:
    parts = [test["title"]]
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


def cluster(
    tests: list[dict],
    threshold: float,
    min_size: int,
    max_size: int,
    lsh_threshold: float,
    num_perm: int,
    embedder_name: str,
) -> list[list[str]]:
    texts = [test_text(t) for t in tests]

    # Stage 1: MinHash-LSH candidate pairs.
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    minhashes: list[MinHash] = []
    for i, text in enumerate(texts):
        mh = build_minhash(text, num_perm)
        lsh.insert(str(i), mh)
        minhashes.append(mh)

    # Stage 2: TF-IDF cosine filter on candidate pairs.
    embedder = get_embedder(embedder_name)
    matrix = embedder.fit_transform(texts)

    uf = UnionFind(len(tests))
    seen: set[tuple[int, int]] = set()
    for i in range(len(tests)):
        for cand in lsh.query(minhashes[i]):
            j = int(cand)
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            sim = cosine_similarity(matrix[a], matrix[b])[0, 0]
            if sim >= threshold:
                uf.union(a, b)

    groups: dict[int, list[int]] = {}
    for idx in range(len(tests)):
        groups.setdefault(uf.find(idx), []).append(idx)

    clusters = [g for g in groups.values() if len(g) >= min_size]

    # Split oversized clusters by chopping into chunks (keeps batch-friendly size).
    sized: list[list[int]] = []
    for g in clusters:
        if len(g) <= max_size:
            sized.append(g)
        else:
            for k in range(0, len(g), max_size):
                chunk = g[k : k + max_size]
                if len(chunk) >= min_size:
                    sized.append(chunk)

    return [[tests[i]["id"] for i in g] for g in sized]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tests", type=Path)
    ap.add_argument("--out", type=Path, default=Path("clusters.json"))
    ap.add_argument("--threshold", type=float, default=0.75, help="TF-IDF cosine threshold")
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=20)
    ap.add_argument("--lsh-threshold", type=float, default=0.5)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--embedder", default="tfidf")
    args = ap.parse_args()

    tests = json.loads(args.tests.read_text(encoding="utf-8"))
    clusters = cluster(
        tests,
        args.threshold,
        args.min_size,
        args.max_size,
        args.lsh_threshold,
        args.num_perm,
        args.embedder,
    )
    args.out.write_text(json.dumps(clusters, ensure_ascii=False, indent=2), encoding="utf-8")

    sizes = [len(c) for c in clusters]
    total = sum(sizes)
    mean_size = float(np.mean(sizes)) if sizes else 0.0
    print(f"{len(clusters)} clusters, {total} tests in clusters (mean size {mean_size:.1f}) → {args.out}")


if __name__ == "__main__":
    main()
