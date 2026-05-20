#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_npc_llm.product.agent import GameAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local rule-policy NPC demo.")
    parser.add_argument("--npc", default="mika")
    parser.add_argument("--session-id", default="cli")
    parser.add_argument("--message", default="The tide engine pressure is climbing. What should I do?")
    args = parser.parse_args()

    agent = GameAgent.demo()
    result = agent.chat(args.npc, args.message, args.session_id)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
