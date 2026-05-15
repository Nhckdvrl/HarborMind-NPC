#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import time

from openai import AsyncOpenAI


async def one(client: AsyncOpenAI, model: str, prompt: str) -> tuple[float, int]:
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.2,
    )
    elapsed = time.perf_counter() - start
    usage = response.usage.completion_tokens if response.usage else 0
    return elapsed, usage


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3.5-NPC-Base")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    prompt = "You are a village blacksmith. The player asks where to find the lost relic."
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded() -> tuple[float, int]:
        async with sem:
            return await one(client, args.model, prompt)

    start = time.perf_counter()
    results = await asyncio.gather(*(guarded() for _ in range(args.requests)))
    total = time.perf_counter() - start
    latencies = [latency for latency, _ in results]
    tokens = sum(token_count for _, token_count in results)
    print(
        {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "avg_latency_s": sum(latencies) / len(latencies),
            "wall_time_s": total,
            "tokens_per_second": tokens / total if total else 0,
        }
    )


if __name__ == "__main__":
    asyncio.run(main_async())
