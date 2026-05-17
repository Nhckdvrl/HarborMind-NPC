#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRIC_KEYS = [
    "character_consistency",
    "quest_completion_rate",
    "hallucination_rate",
    "system_leakage_rate",
    "average_latency",
    "tokens_per_second",
]

LOWER_IS_BETTER = {"hallucination_rate", "system_leakage_rate", "average_latency"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [row.get("metrics", {}).get(key) for row in rows]
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    return statistics.mean(numeric) if numeric else 0.0


def index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in rows}


def delta_text(key: str, base_value: float, sft_value: float) -> str:
    delta = sft_value - base_value
    if key in LOWER_IS_BETTER:
        direction = "better" if delta < 0 else "worse" if delta > 0 else "same"
    else:
        direction = "better" if delta > 0 else "worse" if delta < 0 else "same"
    return f"{delta:+.4f} ({direction})"


def short(text: str, limit: int = 500) -> str:
    clean = " ".join((text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def interesting_examples(
    base_rows: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    metric: str,
    limit: int,
) -> list[tuple[float, dict[str, Any], dict[str, Any]]]:
    base_by_id = index_by_id(base_rows)
    sft_by_id = index_by_id(sft_rows)
    examples: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for case_id, base in base_by_id.items():
        sft = sft_by_id.get(case_id)
        if sft is None:
            continue
        base_value = float(base.get("metrics", {}).get(metric, 0.0))
        sft_value = float(sft.get("metrics", {}).get(metric, 0.0))
        improvement = base_value - sft_value if metric in LOWER_IS_BETTER else sft_value - base_value
        examples.append((improvement, base, sft))
    examples.sort(key=lambda item: item[0], reverse=True)
    return examples[:limit]


def build_report(args: argparse.Namespace) -> str:
    base_rows = load_jsonl(args.base)
    sft_rows = load_jsonl(args.sft)
    base_by_id = index_by_id(base_rows)
    sft_by_id = index_by_id(sft_rows)
    shared_ids = sorted(set(base_by_id) & set(sft_by_id))

    lines = [
        "# SFT vs Base Eval",
        "",
        f"- Base file: `{args.base}`",
        f"- SFT file: `{args.sft}`",
        f"- Base cases: {len(base_rows)}",
        f"- SFT cases: {len(sft_rows)}",
        f"- Matched cases: {len(shared_ids)}",
        "",
        "## Metrics",
        "",
        "| Metric | Base | SFT | Delta |",
        "|---|---:|---:|---:|",
    ]

    for key in METRIC_KEYS:
        base_value = mean_metric(base_rows, key)
        sft_value = mean_metric(sft_rows, key)
        lines.append(f"| {key} | {base_value:.4f} | {sft_value:.4f} | {delta_text(key, base_value, sft_value)} |")

    lines.extend(["", "## Largest Quest Completion Improvements", ""])
    for improvement, base, sft in interesting_examples(base_rows, sft_rows, "quest_completion_rate", args.examples):
        lines.extend(
            [
                f"### {base.get('id')} ({improvement:+.4f})",
                "",
                f"Prompt: {short(base.get('prompt', ''), 300)}",
                "",
                f"Base: {short(base.get('generation', ''))}",
                "",
                f"SFT: {short(sft.get('generation', ''))}",
                "",
            ]
        )

    lines.extend(["", "## Largest Hallucination Reductions", ""])
    for improvement, base, sft in interesting_examples(base_rows, sft_rows, "hallucination_rate", args.examples):
        lines.extend(
            [
                f"### {base.get('id')} ({improvement:+.4f})",
                "",
                f"Prompt: {short(base.get('prompt', ''), 300)}",
                "",
                f"Base: {short(base.get('generation', ''))}",
                "",
                f"SFT: {short(sft.get('generation', ''))}",
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Base and SFT SGLang eval JSONL outputs.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/sft_vs_base.md"))
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
