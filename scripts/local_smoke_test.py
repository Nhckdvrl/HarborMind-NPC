#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(args, cwd=ROOT, env={**env}, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(result.returncode)
    print(result.stdout.strip())


def main() -> None:
    py = sys.executable
    run([py, "scripts/download_data.py", "--dry-run"])
    run([py, "scripts/validate_jsonl.py", "data/processed/sft_train.jsonl", "--schema", "sft"])
    run([py, "scripts/validate_jsonl.py", "data/processed/grpo_prompts.jsonl", "--schema", "grpo"])
    run([py, "scripts/validate_jsonl.py", "data/processed/eval_cases.jsonl", "--schema", "eval"])
    run([py, "scripts/validate_lm_eval_task.py"])
    run([py, "scripts/run_agent_demo.py"])


if __name__ == "__main__":
    main()
