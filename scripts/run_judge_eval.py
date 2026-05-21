#!/usr/bin/env python3
"""
LLM-as-judge evaluation across system generations.

Reads per-case generation files from run_model_eval.py, scores each
on 5 CharacterEval/RoleLLM dimensions (absolute 1-5), and produces
pairwise win-rate comparisons between all system pairs.

Required env vars:
  JUDGE_BASE_URL   e.g. https://api.openai.com/v1
  JUDGE_API_KEY    your API key
  JUDGE_MODEL      e.g. gpt-4o  (also settable via --model)

Example:
  JUDGE_BASE_URL=https://api.openai.com/v1 \\
  JUDGE_API_KEY=sk-... \\
  JUDGE_MODEL=gpt-4o \\
  python scripts/run_judge_eval.py \\
    reports/model_eval_base.json \\
    reports/model_eval_sft.json \\
    reports/model_eval_orpo.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_npc_llm.eval.judge import (
    aggregate_absolute_scores,
    aggregate_pairwise,
    score_absolute,
    score_pairwise,
)


def load_eval_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval_suite(path: Path) -> dict[str, dict]:
    """Return case_id → case dict for quick lookup."""
    cases: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            cases[c["id"]] = c
    return cases


def run_absolute(
    system_data: dict[str, Any],
    suite: dict[str, dict],
    model: str,
    concurrency: int,
    limit: int,
) -> dict[str, Any]:
    name = system_data["name"]
    per_case = system_data["per_case"]
    if limit:
        per_case = per_case[:limit]

    results: list[dict[str, Any]] = []

    def _score_one(entry: dict) -> dict:
        case_id = entry.get("id", "")
        case = suite.get(case_id, {})
        if not case:
            return {"id": case_id, "skipped": True}
        scores_info = score_absolute(case, entry["generation"], model=model)
        return {
            "id": case_id,
            "category": entry.get("category", case.get("category", "general")),
            **scores_info,
        }

    print(f"  Running absolute judge for '{name}' ({len(per_case)} cases, concurrency={concurrency})")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_score_one, e): e for e in per_case}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            results.append(result)
            if i % 10 == 0:
                print(f"    {i}/{len(per_case)} scored")

    agg = aggregate_absolute_scores(results)
    return {"name": name, "aggregate": agg, "per_case": results}


def run_pairwise(
    sys_a: dict[str, Any],
    sys_b: dict[str, Any],
    suite: dict[str, dict],
    model: str,
    concurrency: int,
    limit: int,
) -> dict[str, Any]:
    name_a = sys_a["name"]
    name_b = sys_b["name"]

    # Build lookup: case_id → generation for each system
    gen_a = {e["id"]: e["generation"] for e in sys_a["per_case"]}
    gen_b = {e["id"]: e["generation"] for e in sys_b["per_case"]}
    shared_ids = sorted(set(gen_a) & set(gen_b))
    if limit:
        shared_ids = shared_ids[:limit]

    pair_results: list[dict[str, Any]] = []

    def _pair_one(case_id: str) -> dict:
        case = suite.get(case_id, {})
        if not case:
            return {"id": case_id, "skipped": True}
        result = score_pairwise(case, gen_a[case_id], gen_b[case_id], name_a, name_b, model=model)
        return {"id": case_id, "category": case.get("category", "general"), **result}

    print(f"  Pairwise {name_a} vs {name_b} ({len(shared_ids)} cases)")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_pair_one, cid): cid for cid in shared_ids}
        for fut in as_completed(futures):
            pair_results.append(fut.result())

    agg = aggregate_pairwise(
        [r for r in pair_results if not r.get("skipped")], name_a, name_b
    )
    return {
        "system_a": name_a,
        "system_b": name_b,
        "aggregate": agg,
        "per_case": pair_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-judge evaluation on model generation files.")
    parser.add_argument("eval_files", nargs="+", type=Path,
                        help="model_eval_*.json files from run_model_eval.py")
    parser.add_argument("--eval-suite", type=Path,
                        default=Path("data/processed/eval_suite.jsonl"),
                        help="Discriminating eval suite JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL", "gpt-4o"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Cap cases per system (0=all)")
    parser.add_argument("--skip-pairwise", action="store_true",
                        help="Only run absolute scoring, skip pairwise comparisons")
    args = parser.parse_args()

    if "JUDGE_API_KEY" not in os.environ:
        raise SystemExit("JUDGE_API_KEY env var is required.")

    suite = load_eval_suite(args.eval_suite)
    systems = [load_eval_file(p) for p in args.eval_files]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Absolute scoring per system
    print("=== Absolute Scoring ===")
    absolute_results: list[dict[str, Any]] = []
    for sys_data in systems:
        result = run_absolute(sys_data, suite, args.model, args.concurrency, args.limit)
        absolute_results.append(result)
        out = args.output_dir / f"judge_absolute_{sys_data['name']}.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  → {out} | agg: {result['aggregate']}")

    # Pairwise comparisons (all pairs)
    if not args.skip_pairwise and len(systems) > 1:
        print("=== Pairwise Comparisons ===")
        pairwise_results: list[dict[str, Any]] = []
        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                pw = run_pairwise(systems[i], systems[j], suite, args.model, args.concurrency, args.limit)
                pairwise_results.append(pw)
                na, nb = pw["system_a"], pw["system_b"]
                out = args.output_dir / f"judge_pairwise_{na}_vs_{nb}.json"
                out.write_text(json.dumps(pw, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  → {out} | {pw['aggregate']}")

    # Combined summary
    summary = {
        "model": args.model,
        "eval_suite": str(args.eval_suite),
        "systems": [r["name"] for r in absolute_results],
        "absolute": {r["name"]: r["aggregate"] for r in absolute_results},
    }
    if not args.skip_pairwise and len(systems) > 1:
        summary["pairwise"] = [
            {"a": pw["system_a"], "b": pw["system_b"], **pw["aggregate"]}
            for pw in pairwise_results
        ]

    out = args.output_dir / "judge_eval_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary: {out}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
