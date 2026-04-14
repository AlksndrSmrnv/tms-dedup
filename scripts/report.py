#!/usr/bin/env python3
"""Render a markdown report from verified clusters.

Grouping hierarchy (v2):
  Top-level = algorithm `label` (duplicate_full / duplicate_partial /
              same_intent_diff_flow / suspicious_copypaste)
  Each group shows LLM `severity` + `kind` plus per-pair sim_title/sim_steps.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


# Ordered for output: duplicates first, then suspicious copy-paste (separate).
LABEL_ORDER = [
    "duplicate_full",
    "duplicate_partial",
    "same_intent_diff_flow",
    "suspicious_copypaste",
]
LABEL_TITLE = {
    "duplicate_full": "Полные дубликаты (по алгоритму)",
    "duplicate_partial": "Частичные пересечения",
    "same_intent_diff_flow": "Одинаковое намерение, разный флоу",
    "suspicious_copypaste": "Подозрение на copy-paste шагов",
}
LABEL_HINT = {
    "duplicate_full": "Высокая близость по названиям и по шагам. Кандидаты на слияние.",
    "duplicate_partial": "Умеренная близость. Требуют ручного ревью перед слиянием.",
    "same_intent_diff_flow": "Названия почти идентичны, шаги существенно разные — скорее всего разные сценарии одной проверки.",
    "suspicious_copypaste": "Шаги идентичны, но названия различаются осмысленным уточнением. Обычно это ошибка копипаста шагов — НЕ удалять, а чинить шаги.",
}

SEVERITY_BADGE = {
    "full": "🔴 full",
    "partial": "🟡 partial",
    "none": "⚪ none",
    "unknown": "❔ unknown",
}


def render_pair_metrics_table(pair_metrics: list[dict]) -> list[str]:
    if not pair_metrics:
        return []
    out = ["| Пара | sim_title | sim_steps | disambig-conflict |", "|---|---|---|---|"]
    for p in pair_metrics:
        conflict = "⚠️ да" if p.get("disambig_conflict") else ""
        out.append(f"| `{p['a']}` ↔ `{p['b']}` | {p['sim_title']} | {p['sim_steps']} | {conflict} |")
    out.append("")
    return out


def render_group(g: dict, tests_by_id: dict) -> str:
    sev = g.get("severity") or "unknown"
    kind = g.get("kind") or "unknown"
    lines = [f"### Группа `{g['group_id']}` — {SEVERITY_BADGE.get(sev, sev)} · kind={kind}"]
    if g.get("common_check"):
        lines.append(f"**Что проверяют:** {g['common_check']}")
    if g.get("master_id"):
        lines.append(f"**Мастер:** `{g['master_id']}`")
    if g.get("drop_ids"):
        lines.append(f"**К удалению:** {', '.join(f'`{x}`' for x in g['drop_ids'])}")
    if g.get("rationale"):
        lines.append(f"_{g['rationale']}_")

    lines.append("")
    lines.extend(render_pair_metrics_table(g.get("pair_metrics", [])))

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

    by_label: dict[str, list[dict]] = {k: [] for k in LABEL_ORDER}
    for g in verified:
        lbl = g.get("label") or "unknown"
        by_label.setdefault(lbl, []).append(g)

    label_counts = Counter(g.get("label") or "unknown" for g in verified)
    sev_counts = Counter(g.get("severity") or "unknown" for g in verified)
    kind_counts = Counter(g.get("kind") or "unknown" for g in verified)
    # Only count drop_ids from groups the LLM marked as real duplicates.
    drop_total = sum(
        len(g.get("drop_ids") or [])
        for g in verified
        if (g.get("kind") or "") == "duplicate"
    )

    out = ["# TMS Test Deduplication Report", ""]
    out.append("## Summary")
    out.append(f"- Всего групп: **{len(verified)}**")
    out.append("")
    out.append("**По алгоритмическим меткам:**")
    for lbl in LABEL_ORDER:
        if label_counts.get(lbl):
            out.append(f"- {LABEL_TITLE[lbl]}: **{label_counts[lbl]}**")
    out.append("")
    out.append("**По вердикту LLM (severity):**")
    for sev in ["full", "partial", "none", "unknown"]:
        if sev_counts.get(sev):
            out.append(f"- {SEVERITY_BADGE[sev]}: **{sev_counts[sev]}**")
    out.append("")
    out.append("**По виду (kind):**")
    for kind in ["duplicate", "copypaste_suspicion", "distinct", "unknown"]:
        if kind_counts.get(kind):
            out.append(f"- {kind}: **{kind_counts[kind]}**")
    out.append("")
    out.append(f"- Тестов рекомендовано к удалению (только kind=duplicate): **{drop_total}**")
    out.append("")

    # Sections ordered per LABEL_ORDER; any other label at the end.
    ordered_labels = LABEL_ORDER + [l for l in by_label if l not in LABEL_ORDER]
    for lbl in ordered_labels:
        groups = by_label.get(lbl, [])
        if not groups:
            continue
        out.append(f"## {LABEL_TITLE.get(lbl, lbl)} ({len(groups)})")
        if lbl in LABEL_HINT:
            out.append(f"_{LABEL_HINT[lbl]}_")
        out.append("")
        for g in groups:
            out.append(render_group(g, tests_by_id))

    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
