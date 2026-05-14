#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine eval JSON files into a compact comparison report.")
    parser.add_argument("--inputs", nargs="+", required=True, help="name=path pairs")
    parser.add_argument("--output", type=Path, default=Path("reports/comparison.md"))
    args = parser.parse_args()

    rows = []
    for item in args.inputs:
        name, path = item.split("=", 1)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        results = payload.get("results", payload)
        task = results.get("game_npc_bench", results)
        rows.append((name, task))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GameNPCBench Comparison",
        "",
        "| Model | Character | Quest | Hallucination | Leakage | Latency | Tok/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in rows:
        lines.append(
            "| {name} | {character:.3f} | {quest:.3f} | {hallucination:.3f} | {leakage:.3f} | {latency:.3f} | {tps:.3f} |".format(
                name=name,
                character=float(metrics.get("character_consistency", 0)),
                quest=float(metrics.get("quest_completion_rate", 0)),
                hallucination=float(metrics.get("hallucination_rate", 0)),
                leakage=float(metrics.get("system_leakage_rate", 0)),
                latency=float(metrics.get("average_latency", 0)),
                tps=float(metrics.get("tokens_per_second", 0)),
            )
        )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
