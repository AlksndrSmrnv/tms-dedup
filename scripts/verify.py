#!/usr/bin/env python3
"""Prepare LLM prompts for candidate clusters and aggregate LLM responses.

This script does NOT call any LLM. It is a pure orchestrator: the qwen CLI
agent (or a human) must feed each prompt to the model and drop the JSON
reply into responses_out/. Running this script is idempotent — re-run after
dropping responses and it will aggregate whatever is present.

Flow:
  1. First run: reads clusters.json + tests.json, writes prompts_out/batch_NNN.txt.
  2. Agent feeds each prompt to the LLM, saves reply as responses_out/batch_NNN.json.
  3. Second run: same command, now aggregates responses into clusters_verified.json.
"""
import argparse
import json
import sys
from pathlib import Path


def approx_tokens(text: str) -> int:
    # Rough heuristic: 1 token ≈ 3 chars for mixed ru/en. Good enough for budgeting.
    return max(1, len(text) // 3)


def render_test(test: dict) -> str:
    lines = [f"[id={test['id']}] {test['title']}"]
    for i, step in enumerate(test.get("steps", []), 1):
        action = step.get("action") or ""
        expected = step.get("expected") or ""
        lines.append(f"  {i}. {action} → {expected}")
    return "\n".join(lines)


def render_pair_metrics(pair_metrics: list[dict]) -> str:
    if not pair_metrics:
        return ""
    lines = ["Метрики пар (sim_title / sim_steps):"]
    for p in pair_metrics:
        flag = " [disambig-conflict]" if p.get("disambig_conflict") else ""
        lines.append(f"  {p['a']} ↔ {p['b']}: title={p['sim_title']}, steps={p['sim_steps']}{flag}")
    return "\n".join(lines)


def render_group(cluster: dict, tests_by_id: dict[str, dict]) -> str:
    header = f"--- GROUP {cluster['group_id']} (label={cluster['label']}) ---"
    metrics = render_pair_metrics(cluster.get("pair_metrics", []))
    body = "\n\n".join(
        render_test(tests_by_id[tid]) for tid in cluster["test_ids"] if tid in tests_by_id
    )
    parts = [header]
    if metrics:
        parts.append(metrics)
    parts.append(body)
    return "\n".join(parts)


def pack_batches(clusters: list[dict], tests_by_id: dict, prompt_header: str, budget: int) -> list[dict]:
    """Greedy packing of clusters into batches under the token budget."""
    batches: list[dict] = []
    current: list[dict] = []
    current_tokens = approx_tokens(prompt_header)
    reserve = 4000  # reserve for model answer

    for cluster in clusters:
        rendered = render_group(cluster, tests_by_id)
        t = approx_tokens(rendered)
        if current and current_tokens + t + reserve > budget:
            batches.append({"groups": current})
            current = []
            current_tokens = approx_tokens(prompt_header)
        current.append(
            {
                "group_id": cluster["group_id"],
                "label": cluster["label"],
                "test_ids": cluster["test_ids"],
                "pair_metrics": cluster.get("pair_metrics", []),
                "rendered": rendered,
            }
        )
        current_tokens += t

    if current:
        batches.append({"groups": current})
    return batches


def build_prompt(header: str, batch: dict) -> str:
    body = "\n\n".join(g["rendered"] for g in batch["groups"])
    return f"{header}\n{body}\n\n=== КОНЕЦ ГРУПП ==="


def parse_response(raw: str, resp_path: Path) -> dict:
    """Parse LLM JSON reply; salvage between outermost braces if wrapped."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        print(f"warning: could not parse response {resp_path}, skipping batch", file=sys.stderr)
        return {"results": []}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clusters", type=Path)
    ap.add_argument("tests", type=Path)
    ap.add_argument("--out", type=Path, default=Path("clusters_verified.json"))
    ap.add_argument("--prompts-dir", type=Path, default=Path("prompts_out"))
    ap.add_argument("--responses-dir", type=Path, default=Path("responses_out"))
    ap.add_argument(
        "--prompt-template",
        type=Path,
        default=Path(__file__).parent.parent / "prompts" / "verify_cluster.txt",
    )
    ap.add_argument("--batch-tokens", type=int, default=100000)
    args = ap.parse_args()

    clusters = json.loads(args.clusters.read_text(encoding="utf-8"))
    tests = json.loads(args.tests.read_text(encoding="utf-8"))
    tests_by_id = {t["id"]: t for t in tests}
    header = args.prompt_template.read_text(encoding="utf-8")

    batches = pack_batches(clusters, tests_by_id, header, args.batch_tokens)
    args.prompts_dir.mkdir(parents=True, exist_ok=True)
    args.responses_dir.mkdir(parents=True, exist_ok=True)

    # Always (re)write prompts — cheap and keeps them in sync with clusters.json.
    for i, batch in enumerate(batches):
        prompt = build_prompt(header, batch)
        (args.prompts_dir / f"batch_{i:03d}.txt").write_text(prompt, encoding="utf-8")

    # Aggregate any responses already present.
    verified: list[dict] = []
    missing = 0
    for i, batch in enumerate(batches):
        resp_path = args.responses_dir / f"batch_{i:03d}.json"
        if not resp_path.exists():
            missing += 1
            continue
        parsed = parse_response(resp_path.read_text(encoding="utf-8"), resp_path)
        results = {r["group_id"]: r for r in parsed.get("results", [])}
        for g in batch["groups"]:
            r = results.get(g["group_id"], {})
            verified.append(
                {
                    "group_id": g["group_id"],
                    "label": g["label"],
                    "test_ids": g["test_ids"],
                    "pair_metrics": g.get("pair_metrics", []),
                    "severity": r.get("severity") or "unknown",
                    "kind": r.get("kind") or "unknown",
                    "common_check": r.get("common_check", ""),
                    "master_id": r.get("master_id"),
                    "drop_ids": r.get("drop_ids") or [],
                    "rationale": r.get("rationale", ""),
                }
            )

    args.out.write_text(json.dumps(verified, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"prepared {len(batches)} prompt(s) in {args.prompts_dir}")
    print(f"aggregated {len(verified)} verified group(s) → {args.out} (missing responses: {missing})")
    if missing:
        print(
            f"next step: feed each prompts_out/batch_NNN.txt to qwen, "
            f"save the JSON reply to {args.responses_dir}/batch_NNN.json, then rerun this script.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
