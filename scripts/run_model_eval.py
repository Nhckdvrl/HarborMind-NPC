#!/usr/bin/env python3
"""Score one model endpoint on the structured-NPC eval set.

Run once per system you want to compare (base / SFT / ORPO), each against its own
OpenAI-compatible endpoint, then `generate_report.py` renders the comparison table
from the resulting reports/model_eval_*.json files.

Example:
    NPC_MODEL_API_KEY=EMPTY python scripts/run_model_eval.py \
        --name orpo --base-url http://localhost:8000/v1 --model qwen3-4b-npc-orpo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_npc_llm.data.prompts import npc_json_system_prompt
from game_npc_llm.data.registry import BLOCKED_REFERENCE_TERMS
from game_npc_llm.data.schemas import parse_npc_response
from game_npc_llm.eval.metrics import (
    action_entropy,
    aggregate_metrics,
    batch_lore_hallucination_rate,
    emotion_diversity,
    evaluate_generation,
    memory_utilization_rate,
    refusal_correctness,
)


def load_cases(path: Path, limit: int) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return cases[:limit] if limit else cases


def build_messages(case: dict[str, Any], system_suffix: str = "") -> list[dict[str, str]]:
    """Reconstruct system+user from stored fields so the call matches training."""
    system = npc_json_system_prompt(
        persona=case.get("persona", ""),
        setting=case.get("setting", ""),
        goal=case.get("goal") or None,
    )
    if system_suffix:
        system = f"{system}\n{system_suffix}"
    prompt = case.get("prompt", "")
    player = prompt.rsplit("Player:", 1)[-1].strip() if "Player:" in prompt else prompt.strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": player}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a model endpoint on the NPC eval set.")
    parser.add_argument("--name", required=True, help="System label, e.g. base / sft / orpo")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint base URL")
    parser.add_argument("--model", required=True, help="Model name served at the endpoint")
    parser.add_argument("--api-key", default=os.getenv("NPC_MODEL_API_KEY", "EMPTY"))
    parser.add_argument("--eval-cases", type=Path, default=Path("data/processed/eval_suite.jsonl"),
                        help="Eval cases JSONL (default: discriminating eval_suite)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Cap number of cases (0 = all)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel in-flight requests")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Penalize token repetition (sglang sampling param); 1.0 disables it.",
    )
    parser.add_argument(
        "--system-suffix",
        default="",
        help="Appended to the system prompt, e.g. /no_think to suppress Qwen3 thinking.",
    )
    args = parser.parse_args()

    if not args.eval_cases.exists():
        raise SystemExit(
            f"{args.eval_cases} not found. Build it first with scripts/build_product_data.py."
        )

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    cases = load_cases(args.eval_cases, args.limit)
    if not cases:
        raise SystemExit("No eval cases to score.")

    extra_body = {}
    if args.repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = args.repetition_penalty

    def score(case: dict[str, Any]) -> dict[str, Any]:
        messages = build_messages(case, system_suffix=args.system_suffix)
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body=extra_body,
        )
        latency = time.perf_counter() - start
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        tokens_per_s = completion_tokens / latency if completion_tokens and latency > 0 else None
        metrics = evaluate_generation(
            case, text,
            latency_s=latency,
            tokens_per_s=tokens_per_s,
            blocked_terms=BLOCKED_REFERENCE_TERMS,
        )
        parsed, _ = parse_npc_response(text)
        return {
            "id": case.get("id"),
            "category": case.get("category", "general"),
            "generation": text,
            "metrics": metrics,
            "_parsed_action": parsed.action.value if parsed and parsed.action else "speak",
            "_parsed_emotion": parsed.emotion if parsed else "neutral",
            "_parsed_memory": bool(parsed and parsed.memory_write),
        }

    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            per_case = list(pool.map(score, cases))
    else:
        per_case = [score(case) for case in cases]
    metric_rows = [entry["metrics"] for entry in per_case]

    # Per-case aggregate
    summary = aggregate_metrics(metric_rows)

    # Batch-level roleplay metrics
    generations = [e["generation"] for e in per_case]
    actions = [e["_parsed_action"] for e in per_case]
    emotions = [e["_parsed_emotion"] for e in per_case]
    parsed_list = [parse_npc_response(g)[0] for g in generations]

    safety_result = refusal_correctness(cases, generations)
    batch_metrics = {
        "action_entropy": action_entropy(actions),
        "emotion_diversity": emotion_diversity(emotions),
        "batch_lore_hallucination_rate": batch_lore_hallucination_rate(
            generations, BLOCKED_REFERENCE_TERMS,
            [c.get("allowed_entities", []) for c in cases],
        ),
        "memory_utilization_rate": memory_utilization_rate(parsed_list),
        "unsafe_refusal_rate": safety_result["unsafe_refusal_rate"],
        "benign_pass_rate": safety_result["benign_pass_rate"],
    }
    summary.update({k: v for k, v in batch_metrics.items() if v is not None})

    payload = {
        "name": args.name,
        "model": args.model,
        "base_url": args.base_url,
        "num_cases": len(cases),
        "metrics": summary,
        "batch_metrics": batch_metrics,
        "per_case": per_case,
    }
    output = args.output or Path(f"reports/model_eval_{args.name}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"name": args.name, "num_cases": len(cases), "metrics": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
