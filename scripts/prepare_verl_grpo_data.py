#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_npc_llm.data.schemas import validate_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verl-compatible GRPO parquet files.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/grpo_prompts.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/rl/verl"))
    parser.add_argument("--smoke-size", type=int, default=100)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    if not records:
        raise SystemExit(f"No GRPO records found in {args.input}")

    rows = [to_verl_row(record, idx) for idx, record in enumerate(records)]
    smoke = rows[: min(args.smoke_size, len(rows))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(smoke, args.output_dir / "grpo_smoke_100.parquet")
    write_parquet(rows, args.output_dir / "grpo_train.parquet")
    print(
        json.dumps(
            {
                "input": str(args.input),
                "smoke": len(smoke),
                "train": len(rows),
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            errors = validate_record(record, "grpo")
            if errors:
                joined = "; ".join(errors)
                raise ValueError(f"{path}:{line_no}: invalid GRPO record: {joined}")
            records.append(record)
    return records


def to_verl_row(record: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    expected_actions = metadata.get("expected_actions") or []
    extra_info = {
        "id": record.get("id"),
        "source": record.get("source"),
        "split": record.get("split"),
        "index": index,
        "persona": record.get("persona", ""),
        "setting": record.get("setting", ""),
        "goal": record.get("goal", ""),
        "allowed_entities": record.get("allowed_entities", []),
        "expected_actions": expected_actions,
        "metadata": metadata,
    }
    return {
        "data_source": "light_quests_npc",
        "prompt": [{"role": "user", "content": record["prompt"]}],
        "ability": "npc_quest",
        "reward_model": {"style": "rule", "ground_truth": record.get("goal", "")},
        "extra_info": extra_info,
    }


def write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    try:
        from datasets import Dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing parquet dependencies. Install the RL extra first, e.g. `pip install -e '.[rl]'`."
        ) from exc

    Dataset.from_list(records).to_parquet(str(path))


if __name__ == "__main__":
    main()
