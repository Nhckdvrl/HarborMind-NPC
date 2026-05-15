#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GRPO prompts for slime.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/grpo_prompts.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/rl/grpo_prompts.jsonl"))
    args = parser.parse_args()

    records = [to_slime_record(record) for record in read_jsonl(args.input)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} slime prompt records to {args.output}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_slime_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    metadata.update(
        {
            "id": record.get("id"),
            "source": record.get("source"),
            "persona": record.get("persona", ""),
            "setting": record.get("setting", ""),
            "goal": record.get("goal", ""),
            "allowed_entities": record.get("allowed_entities", []),
        }
    )
    return {
        "prompt": record["prompt"],
        "label": record.get("goal", ""),
        "metadata": metadata,
    }


if __name__ == "__main__":
    main()
