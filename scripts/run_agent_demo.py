#!/usr/bin/env python3
"""Interactive NPC demo with Qwen3 local model support.

Single-turn mode:
    python scripts/run_agent_demo.py --npc mika --message "你好"

Interactive loop mode (no --message):
    NPC_MODEL_URL=http://localhost:30000/v1 python scripts/run_agent_demo.py --npc mika

    Or with explicit args:
    python scripts/run_agent_demo.py --npc mika --model-url http://localhost:30000/v1

Controls during interactive mode:
    reset   — clear session state and start fresh
    quit/q  — exit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_npc_llm.product.agent import GameAgent
from game_npc_llm.product.policy import RulePolicyClient


def _is_llm_active(agent: GameAgent) -> bool:
    return not isinstance(agent.policy, RulePolicyClient)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NPC agent demo.")
    parser.add_argument("--npc", default="mika", help="NPC id to talk to")
    parser.add_argument("--session-id", default="cli")
    parser.add_argument("--message", default=None, help="Single-turn message (omit for interactive loop)")
    parser.add_argument("--model-url", default=None, help="OpenAI-compatible API base URL, e.g. http://localhost:30000/v1")
    parser.add_argument("--model", default=None, help="Model name to request (default: qwen3-4b)")
    args = parser.parse_args()

    # Propagate CLI args to env vars so create_policy_from_env() picks them up.
    if args.model_url:
        os.environ["NPC_MODEL_URL"] = args.model_url
    if args.model:
        os.environ["NPC_MODEL"] = args.model

    agent = GameAgent.demo()
    npc = agent.world.npcs.get(args.npc)
    if npc is None:
        print(f"Unknown NPC '{args.npc}'. Available: {list(agent.world.npcs)}", file=sys.stderr)
        sys.exit(1)

    mode = "LLM" if _is_llm_active(agent) else "rule-based (set NPC_MODEL_URL to enable LLM)"
    npc_name = npc.name

    if args.message:
        # ── Single-turn mode ─────────────────────────────────────────────
        result = agent.chat(args.npc, args.message, args.session_id)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    # ── Interactive loop mode ─────────────────────────────────────────────
    print(f"=== {npc_name} | policy: {mode} ===")
    print("Commands: 'reset' to restart session, 'quit' / 'q' to exit\n")

    while True:
        try:
            player_input = input("Player> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not player_input:
            continue
        if player_input.lower() in ("quit", "exit", "q"):
            break
        if player_input.lower() == "reset":
            agent.reset(args.session_id)
            print(f"[Session reset]\n")
            continue

        result = agent.chat(args.npc, player_input, args.session_id)

        print(f"\n{npc_name}: {result.response.dialogue}")
        if result.visible_events:
            for event in result.visible_events:
                print(f"  [{event}]")
        print()


if __name__ == "__main__":
    main()
