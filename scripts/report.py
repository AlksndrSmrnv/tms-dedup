#!/usr/bin/env python3
"""Render a markdown report from verified clusters."""
import argparse
import json
from collections import Counter
from pathlib import Path


SEVERITY_ORDER = ["full", "partial", "none", "unknown"]
SEVERITY_TITLE = {
    "full": "Полные дубликаты",
    "partial": "Частичные пересечения",
    "none": "Ложные срабатывания",
    "unknown": "Без вердикта",
}


def render_group(g: dict, tests_by_id: dict) -> str:
    lines = [f"### Группа `{g['group_id']}` — {g['severity']}"]
    if g.get("common_check"):
        lines.append(f"**Что проверяют:** {g['common_check']}")
    if g.get("master_id"):
        lines.append(f"**Мастер:** `{g['master_id']}`")
    if g.get("drop_ids"):
        lines.append(f"**К удалению:** {', '.join(f'`{x}`' for x in g['drop_ids'])}")
    if g.get("rationale"):
        lines.append(f"_{g['rationale']}_")

    lines.append("")
    lines.append("| ID | Название |")
    lines.append("|---|---|")
    for tid in g["test_ids"]:
        title = tests_by_id.get(tid, {}).get("title", "?")
        marker = ""
        if tid == g.get("master_id"):
            marker = " ⭐"
        elif tid in (g.get("drop_ids") or []):
            marker = " 🗑"
        lines.append(f"| `{tid}`{marker} | {title} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("verified", type=Path)
    ap.add_argument("tests", type=Path)
    ap.add_argument("--out", type=Path, default=Path("report.md"))
    args = ap.parse_args()

    verified = json.loads(args.verified.read_text(encoding="utf-8"))
    tests = json.loads(args.tests.read_text(encoding="utf-8"))
    tests_by_id = {t["id"]: t for t in tests}

    by_sev: dict[str, list[dict]] = {k: [] for k in SEVERITY_ORDER}
    for g in verified:
        by_sev.setdefault(g["severity"], []).append(g)

    sev_counts = Counter(g["severity"] for g in verified)
    drop_total = sum(len(g.get("drop_ids") or []) for g in verified)

    out = ["# TMS Test Deduplication Report", ""]
    out.append("## Summary")
    out.append(f"- Всего групп: **{len(verified)}**")
    for sev in SEVERITY_ORDER:
        if sev_counts.get(sev):
            out.append(f"- {SEVERITY_TITLE[sev]}: **{sev_counts[sev]}**")
    out.append(f"- Тестов рекомендовано к удалению: **{drop_total}**")
    out.append("")

    for sev in SEVERITY_ORDER:
        groups = by_sev.get(sev, [])
        if not groups:
            continue
        out.append(f"## {SEVERITY_TITLE[sev]} ({len(groups)})")
        out.append("")
        for g in groups:
            out.append(render_group(g, tests_by_id))

    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
