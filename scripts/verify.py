#!/usr/bin/env python3
"""Batch candidate clusters under a token budget and verify via LLM.

Two modes:
  default (agent-driven): write prompts to prompts_out/ and expect the caller
    (qwen CLI / human) to return JSON responses as responses_out/batch_*.json.
  --api: call an OpenAI-compatible endpoint directly.
"""
import argparse
import json
import os
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


def render_group(group_id: str, test_ids: list[str], tests_by_id: dict[str, dict]) -> str:
    header = f"--- GROUP {group_id} ---"
    body = "\n\n".join(render_test(tests_by_id[tid]) for tid in test_ids if tid in tests_by_id)
    return f"{header}\n{body}"


def pack_batches(clusters: list[list[str]], tests_by_id: dict, prompt_header: str, budget: int) -> list[dict]:
    """Greedy packing of clusters into batches under the token budget."""
    batches: list[dict] = []
    current: list[dict] = []
    current_tokens = approx_tokens(prompt_header)
    reserve = 4000  # reserve for model answer

    for idx, cluster in enumerate(clusters):
        gid = f"g{idx}"
        rendered = render_group(gid, cluster, tests_by_id)
        t = approx_tokens(rendered)
        if current and current_tokens + t + reserve > budget:
            batches.append({"groups": current})
            current = []
            current_tokens = approx_tokens(prompt_header)
        current.append({"group_id": gid, "test_ids": cluster, "rendered": rendered})
        current_tokens += t

    if current:
        batches.append({"groups": current})
    return batches


def build_prompt(header: str, batch: dict) -> str:
    body = "\n\n".join(g["rendered"] for g in batch["groups"])
    return f"{header}\n{body}\n\n=== КОНЕЦ ГРУПП ==="


def call_openai_api(prompt: str, model: str, base_url: str, api_key: str) -> str:
    import urllib.request

    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clusters", type=Path)
    ap.add_argument("tests", type=Path)
    ap.add_argument("--out", type=Path, default=Path("clusters_verified.json"))
    ap.add_argument("--prompts-dir", type=Path, default=Path("prompts_out"))
    ap.add_argument("--responses-dir", type=Path, default=Path("responses_out"))
    ap.add_argument("--prompt-template", type=Path, default=Path(__file__).parent.parent / "prompts" / "verify_cluster.txt")
    ap.add_argument("--batch-tokens", type=int, default=100000)
    ap.add_argument("--api", action="store_true", help="Call OpenAI-compatible API directly")
    ap.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--model", default="qwen-coder-next")
    args = ap.parse_args()

    clusters = json.loads(args.clusters.read_text(encoding="utf-8"))
    tests = json.loads(args.tests.read_text(encoding="utf-8"))
    tests_by_id = {t["id"]: t for t in tests}
    header = args.prompt_template.read_text(encoding="utf-8")

    batches = pack_batches(clusters, tests_by_id, header, args.batch_tokens)
    args.prompts_dir.mkdir(parents=True, exist_ok=True)
    args.responses_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(batches):
        prompt = build_prompt(header, batch)
        (args.prompts_dir / f"batch_{i:03d}.txt").write_text(prompt, encoding="utf-8")

    print(f"prepared {len(batches)} batches in {args.prompts_dir}")

    if args.api:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            sys.exit(f"env var {args.api_key_env} is not set")
        for i, batch in enumerate(batches):
            resp_path = args.responses_dir / f"batch_{i:03d}.json"
            if resp_path.exists():
                print(f"batch {i}: cached")
                continue
            prompt = build_prompt(header, batch)
            print(f"batch {i}: calling {args.model}…")
            raw = call_openai_api(prompt, args.model, args.api_base, api_key)
            resp_path.write_text(raw, encoding="utf-8")
    else:
        print(
            f"agent mode: feed each prompts_out/batch_NNN.txt to the LLM, "
            f"save JSON replies to {args.responses_dir}/batch_NNN.json, then rerun this script."
        )

    # Aggregate responses into verified clusters.
    verified = []
    missing = 0
    for i, batch in enumerate(batches):
        resp_path = args.responses_dir / f"batch_{i:03d}.json"
        if not resp_path.exists():
            missing += 1
            continue
        raw = resp_path.read_text(encoding="utf-8")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try to salvage JSON object by finding first "{" and last "}"
            start, end = raw.find("{"), raw.rfind("}")
            try:
                parsed = json.loads(raw[start : end + 1]) if start >= 0 and end > start else {"results": []}
            except json.JSONDecodeError:
                print(f"warning: could not parse response {resp_path}, skipping batch", file=sys.stderr)
                parsed = {"results": []}
        results = {r["group_id"]: r for r in parsed.get("results", [])}
        for g in batch["groups"]:
            r = results.get(g["group_id"], {})
            verified.append(
                {
                    "group_id": g["group_id"],
                    "test_ids": g["test_ids"],
                    "severity": r.get("severity", "unknown"),
                    "common_check": r.get("common_check", ""),
                    "master_id": r.get("master_id"),
                    "drop_ids": r.get("drop_ids", []),
                    "rationale": r.get("rationale", ""),
                }
            )

    args.out.write_text(json.dumps(verified, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(verified)} verified groups → {args.out} (missing batches: {missing})")


if __name__ == "__main__":
    main()
