#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_npc_llm.eval.tasks.game_npc_bench import GameNPCBench


def main() -> None:
    task = GameNPCBench()
    docs = list(task.validation_docs())
    print({"task": "game_npc_bench", "docs": len(docs), "ok": True})


if __name__ == "__main__":
    main()
