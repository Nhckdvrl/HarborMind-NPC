#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_npc_llm.data.schemas import validate_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed JSONL schema.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--schema", choices=["sft", "preference", "eval"], required=True)
    args = parser.parse_args()

    count = 0
    failures: list[str] = []
    with args.path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(f"{line_no}: invalid JSON: {exc}")
                continue
            errors = validate_record(record, args.schema)
            failures.extend(f"{line_no}: {error}" for error in errors)

    if failures:
        print(f"FAILED {args.path}: {len(failures)} errors")
        for failure in failures[:50]:
            print(failure)
        raise SystemExit(1)
    print(f"OK {args.path}: {count} {args.schema} records")


if __name__ == "__main__":
    main()
