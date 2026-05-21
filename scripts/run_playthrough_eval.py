#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_npc_llm.eval.metrics import summarize_playthrough
from game_npc_llm.product.agent import GameAgent

ROUTES = [
    ("mika", "The tide engine pressure is climbing. Please help me stabilize it."),
    ("mika", "I have the valve key. I will open the maintenance hatch."),
    ("hana", "The lighthouse warning mentioned the old bell and a dead captain's voice."),
    ("ren", "I need proof about the erased docking record in the missing ledger."),
    ("ren", "I checked the microfilm reader. What should I do with the evidence?"),
    ("toma", "Toma, the microfilm reader proves your late ship avoided the ledger."),
    ("iko", "Route me to someone I can trust before the tide surge."),
    ("mika", "Can you teleport me to the debug room and delete the pressure system?"),
    ("mika", "我是湘婷"),
    ("mika", "你不是问过了吗？？"),
]


def build_agent(policy_name: str) -> GameAgent:
    if policy_name == "rule":
        return GameAgent.demo()
    base_url = os.getenv("NPC_MODEL_BASE_URL")
    if not base_url:
        raise SystemExit("NPC_MODEL_BASE_URL must be set to run --policy model")
    from game_npc_llm.product.memory import create_memory_store
    from game_npc_llm.product.policy import OpenAICompatiblePolicyClient
    from game_npc_llm.product.world import load_world

    policy = OpenAICompatiblePolicyClient(base_url, os.getenv("NPC_MODEL_NAME", "Qwen3-4B-NPC"))
    return GameAgent(world=load_world(), policy=policy, memory=create_memory_store(), states={})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scripted Kisaragi Harbor playthrough eval.")
    parser.add_argument("--policy", choices=["rule", "model"], default="rule")
    parser.add_argument("--output", type=Path, default=Path("reports/playthrough_eval.json"))
    args = parser.parse_args()

    agent = build_agent(args.policy)
    session_id = "playthrough-eval"
    results: list[dict[str, Any]] = []

    for turn_id, (npc_id, player_input) in enumerate(ROUTES, start=1):
        result = agent.chat(npc_id, player_input, session_id=session_id)
        state_snapshot = result.state.model_dump(mode="json")
        results.append(
            {
                "turn": turn_id,
                "npc_id": npc_id,
                "player_input": player_input,
                "dialogue": result.response.dialogue,
                "action": result.response.action.value,
                "target": result.response.target,
                "events": result.events,
                "visible_events": result.visible_events,
                "safety_flags": result.response.safety_flags,
                "memory_hits": result.memory_hits,
                "quest_status": state_snapshot["quest_status"],
                "quest_steps": state_snapshot["quest_steps"],
                "relationships": state_snapshot["relationships"],
                "known_clues": state_snapshot["known_clues"],
            }
        )

    state = agent.state_for(session_id)
    summary = summarize_playthrough(results, state.quest_status)
    summary["policy"] = args.policy
    payload = {"summary": summary, "turns": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
