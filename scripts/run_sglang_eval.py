#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_npc_llm.eval.metrics import evaluate_generation


METRIC_KEYS = [
    "character_consistency",
    "quest_completion_rate",
    "hallucination_rate",
    "system_leakage_rate",
    "average_latency",
    "tokens_per_second",
]


def load_cases(path: Path, limit: int | None = None, split: str | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            if split is not None and case.get("split") != split:
                continue
            cases.append(case)
            if limit is not None and len(cases) >= limit:
                break
    return cases


async def generate_one(
    client: AsyncOpenAI,
    model: str,
    case: dict[str, Any],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    disable_thinking: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        extra_body = None
        if disable_thinking:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": case["prompt"]}],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body,
            ),
            timeout=timeout_s,
        )
        latency_s = time.perf_counter() - start
        generation = response.choices[0].message.content or ""
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        tokens_per_s = completion_tokens / latency_s if latency_s and completion_tokens else 0.0
        metrics = evaluate_generation(case, generation, latency_s=latency_s, tokens_per_s=tokens_per_s)
        error = None
    except Exception as exc:
        latency_s = time.perf_counter() - start
        generation = ""
        completion_tokens = 0
        tokens_per_s = 0.0
        metrics = {}
        error = f"{type(exc).__name__}: {exc}"

    return {
        "id": case.get("id"),
        "source": case.get("source"),
        "split": case.get("split"),
        "category": case.get("category"),
        "prompt": case.get("prompt"),
        "reference": case.get("reference"),
        "generation": generation,
        "latency_s": latency_s,
        "completion_tokens": completion_tokens,
        "tokens_per_second": tokens_per_s,
        "metrics": metrics,
        "error": error,
    }


async def run_eval(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = load_cases(args.cases, limit=args.limit, split=args.split)
    if not cases:
        raise SystemExit(f"No eval cases found in {args.cases}.")

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    if not args.skip_health_check:
        try:
            await asyncio.wait_for(client.models.list(), timeout=args.health_timeout)
        except Exception as exc:
            raise SystemExit(
                f"Endpoint is not reachable before eval: {args.base_url}\n"
                f"Error: {type(exc).__name__}: {exc}\n"
                "Start the matching SGLang service and verify `/v1/models` first."
            ) from exc

    sem = asyncio.Semaphore(args.concurrency)
    progress = tqdm(total=len(cases), desc=f"eval {args.model}", unit="case")

    async def guarded(case: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            result = await generate_one(
                client=client,
                model=args.model,
                case=case,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_s=args.timeout,
                disable_thinking=args.disable_thinking,
            )
            progress.update(1)
            return result

    try:
        return await asyncio.gather(*(guarded(case) for case in cases))
    finally:
        progress.close()


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [result for result in results if result.get("error")]
    successes = [result for result in results if not result.get("error")]
    summary: dict[str, Any] = {
        "num_cases": len(results),
        "num_success": len(successes),
        "num_errors": len(errors),
        "error_rate": len(errors) / len(results) if results else 0.0,
    }
    for key in METRIC_KEYS:
        values = [result["metrics"][key] for result in successes if key in result.get("metrics", {})]
        summary[key] = statistics.mean(values) if values else 0.0
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GameNPCBench cases against an SGLang OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--model", required=True, help="Served model name, e.g. Qwen3.5-NPC-Base")
    parser.add_argument("--cases", type=Path, default=Path("data/eval/eval_cases.jsonl"))
    parser.add_argument("--output", type=Path, required=True, help="JSONL output path for per-case generations.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional summary JSON path.")
    parser.add_argument("--split", default=None, help="Optional split filter, e.g. test or validation.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of cases for smoke runs.")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--health-timeout", type=float, default=10.0)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass chat_template_kwargs.enable_thinking=false for Qwen reasoning chat templates.",
    )
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--api-key", default="EMPTY")
    args = parser.parse_args()

    results = asyncio.run(run_eval(args))
    write_jsonl(args.output, results)

    summary = summarize(results)
    summary.update(
        {
            "base_url": args.base_url,
            "model": args.model,
            "cases": str(args.cases),
            "output": str(args.output),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "concurrency": args.concurrency,
            "disable_thinking": args.disable_thinking,
        }
    )

    summary_output = args.summary_output or args.output.with_suffix(".summary.json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
