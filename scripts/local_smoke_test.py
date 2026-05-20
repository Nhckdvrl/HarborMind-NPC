#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    py = sys.executable
    with tempfile.TemporaryDirectory(prefix="game-npc-smoke-") as tmp:
        output_dir = Path(tmp) / "data"
        run([py, "scripts/build_product_data.py", "--dry-run", "--output-dir", str(output_dir)])
        run([py, "scripts/validate_jsonl.py", str(output_dir / "processed/sft_train.jsonl"), "--schema", "sft"])
        run(
            [
                py,
                "scripts/validate_jsonl.py",
                str(output_dir / "processed/preference_train.jsonl"),
                "--schema",
                "preference",
            ]
        )
        run([py, "scripts/validate_jsonl.py", str(output_dir / "processed/eval_cases.jsonl"), "--schema", "eval"])
        run([py, "-m", "pytest"])


if __name__ == "__main__":
    main()
