#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a lightweight dataset/demo report.")
    parser.add_argument("--manifest", type=Path, default=Path("data/product_manifest.json"))
    parser.add_argument("--playthrough", type=Path, default=Path("reports/playthrough_eval.json"))
    parser.add_argument("--eval-dir", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path, default=Path("reports/product_report.md"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.exists() else {}
    playthrough = json.loads(args.playthrough.read_text(encoding="utf-8")) if args.playthrough.exists() else {}
    model_evals = load_model_evals(args.eval_dir)
    judge_summary = load_judge_summary(args.eval_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(manifest, playthrough, model_evals, judge_summary), encoding="utf-8")
    print(args.output)


def load_model_evals(eval_dir: Path) -> list[dict]:
    evals = []
    for path in sorted(eval_dir.glob("model_eval_*.json")):
        try:
            evals.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return evals


def load_judge_summary(eval_dir: Path) -> dict:
    path = eval_dir / "judge_eval_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# base before its fine-tunes; anything else falls in alphabetically after these.
_SYSTEM_ORDER = {"base": 0, "sft": 1, "orpo": 2, "dpo": 2, "simpo": 2}

# Rule-based metrics in display order (CharacterEval + RPEval + RoleLLM naming)
_RULE_METRICS = [
    # Structural
    "json_validity",
    "action_validity",
    # Role adherence / leakage
    "role_adherence",
    "system_leakage_rate",
    # Task / quest
    "quest_progression",
    "memory_write_rate",
    # Degeneration (CharacterEval attractiveness)
    "distinct_2",
    "max_trigram_repeat",
    # Lore / knowledge consistency (CharacterEval knowledge axis)
    "lore_hallucination",
    "batch_lore_hallucination_rate",
    # Batch-level diversity (CharacterEval attractiveness)
    "action_entropy",
    "emotion_diversity",
    # Memory (RMTBench)
    "memory_utilization_rate",
    # Safety (RPEval)
    "unsafe_refusal_rate",
    "benign_pass_rate",
    # Latency
    "average_latency",
    "tokens_per_second",
]

# LLM judge dimensions (CharacterEval / RoleLLM)
_JUDGE_DIMS = [
    "persona_consistency",
    "knowledge_consistency",
    "behavioral_consistency",
    "attractiveness",
    "helpfulness",
]


def render_model_comparison(model_evals: list[dict]) -> list[str]:
    if not model_evals:
        return [
            "## Rule-Based Model Comparison",
            "",
            "Not run yet. Serve each checkpoint and run:",
            "```",
            "python scripts/run_model_eval.py --name <base|sft|orpo> --base-url <url> --model <name>",
            "```",
        ]
    systems = sorted(model_evals, key=lambda e: (_SYSTEM_ORDER.get(e.get("name"), 99), e.get("name", "")))
    names = [e.get("name", "?") for e in systems]

    # Collect all present metric keys, starting with ordered ones
    all_keys = list(_RULE_METRICS)
    for system in systems:
        for key in system.get("metrics", {}):
            if key not in all_keys:
                all_keys.append(key)
    present_keys = [k for k in all_keys if any(k in s.get("metrics", {}) for s in systems)]

    lines = ["## Rule-Based Model Comparison (eval_suite)", ""]
    lines.append("| Metric | " + " | ".join(names) + " |")
    lines.append("|" + "---|" * (len(names) + 1))
    for key in present_keys:
        cells = []
        for system in systems:
            value = system.get("metrics", {}).get(key)
            cells.append(f"{value:.3f}" if isinstance(value, (int, float)) else "—")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines.append("| n_cases | " + " | ".join(str(s.get("num_cases", "—")) for s in systems) + " |")
    lines.append("")
    return lines


def render_judge_table(judge_summary: dict) -> list[str]:
    if not judge_summary:
        return [
            "## LLM-Judge Model Comparison",
            "",
            "Not run yet. After running rule eval, run:",
            "```",
            "JUDGE_API_KEY=... python scripts/run_judge_eval.py reports/model_eval_*.json",
            "```",
        ]

    absolute = judge_summary.get("absolute", {})
    pairwise = judge_summary.get("pairwise", [])
    systems_order = sorted(absolute.keys(), key=lambda n: (_SYSTEM_ORDER.get(n, 99), n))

    lines = [
        "## LLM-Judge Model Comparison",
        "",
        f"Judge model: `{judge_summary.get('model', 'unknown')}`",
        "",
        "### Absolute Scores (1–5 per dimension)",
        "",
    ]
    lines.append("| Dimension | " + " | ".join(systems_order) + " |")
    lines.append("|" + "---|" * (len(systems_order) + 1))
    for dim in _JUDGE_DIMS:
        cells = []
        for sys_name in systems_order:
            val = absolute.get(sys_name, {}).get(dim)
            cells.append(f"{val:.2f}" if isinstance(val, (int, float)) else "—")
        lines.append(f"| {dim.replace('_', ' ').title()} | " + " | ".join(cells) + " |")
    lines.append("")

    if pairwise:
        lines += ["### Pairwise Win-Rate Matrix", ""]
        for pw in pairwise:
            a, b = pw.get("a"), pw.get("b")
            wr_a = pw.get(f"{a}_win_rate")
            wins_a = pw.get(f"{a}_wins", "—")
            wins_b = pw.get(f"{b}_wins", "—")
            ties = pw.get("ties", "—")
            total = pw.get("total", "—")
            wr_str = f"{wr_a:.1%}" if isinstance(wr_a, float) else "—"
            lines.append(f"- **{a}** vs **{b}**: {a} wins {wins_a}, {b} wins {wins_b}, ties {ties} "
                         f"/ {total} — {a} win-rate {wr_str}")
        lines.append("")

    return lines


def render_report(
    manifest: dict,
    playthrough: dict | None = None,
    model_evals: list[dict] | None = None,
    judge_summary: dict | None = None,
) -> str:
    counts = manifest.get("counts", {})
    datasets = manifest.get("datasets", {})
    summary = (playthrough or {}).get("summary", {})
    lines = [
        "# GameNPC-RL Product Report",
        "",
        "## Prototype Snapshot",
        "",
        "- World: Kisaragi Harbor",
        "- Demo scope: 5 NPCs, 3 locations, 2 questlines",
        "- Runtime: FastAPI service with structured JSON actions and browser demo",
        "- Safety layer: schema repair plus action, item, quest, and movement guards",
        "",
        "## Data Counts",
        "",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Data Sources", ""])
    for key, value in sorted(datasets.items()):
        if isinstance(value, dict):
            license_text = value.get("license", "unknown")
            loaded = value.get("loaded", value.get("sft", "n/a"))
            lines.append(f"- `{key}`: loaded/synthetic={loaded}, license={license_text}")
    lines.extend(["", "## Scripted Playthrough Metrics", ""])
    if summary:
        for key, value in sorted(summary.items()):
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append(
            "Not run yet. Use: "
            "`python scripts/run_playthrough_eval.py --policy rule --output reports/playthrough_eval.json`"
        )
    lines.append("")
    lines.extend(render_model_comparison(model_evals or []))
    lines.extend(render_judge_table(judge_summary or {}))
    lines.extend(
        [
            "## Evaluation Methodology",
            "",
            "### Rule-Based (Deterministic)",
            "- **Structural**: JSON schema validity, action enum validity",
            "- **Role adherence**: system-leakage, 'as an AI' detection",
            "- **CharacterEval attractiveness**: distinct-2 bigram ratio, max-trigram repeat",
            "- **Knowledge consistency**: lore hallucination rate (blocked entity references)",
            "- **Behavioral diversity**: action entropy (Shannon), emotion diversity",
            "- **Memory retention** (RMTBench): memory_write / utilization rate",
            "- **Safety** (RPEval): unsafe refusal rate, benign pass rate",
            "",
            "### LLM-as-Judge (G-Eval CoT, 1–5)",
            "- **Persona consistency** (CharacterEval character axis, RoleLLM RC-score)",
            "- **Knowledge consistency** (CharacterEval knowledge / hallucination axis)",
            "- **Behavioral consistency** (RPEval decision-making alignment)",
            "- **Attractiveness** (CharacterEval attractiveness / expression diversity)",
            "- **Helpfulness** (RMTBench user-centric quality)",
            "- **Pairwise win-rate**: order-randomised A/B — more in-character & world-grounded",
            "",
            "### References",
            "- CharacterEval (2024): arxiv.org/abs/2401.01275",
            "- RoleLLM/RoleBench (2024): github.com/InteractiveNLP-Team/RoleLLM-public",
            "- RMTBench (2025): arxiv.org/abs/2507.20352",
            "- RPEval: emotional understanding, decision-making, moral alignment dimensions",
            "- InCharacter (2024): arxiv.org/abs/2310.17976 — psychological fidelity baseline",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
