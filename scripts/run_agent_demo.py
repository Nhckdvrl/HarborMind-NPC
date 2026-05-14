#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_npc_llm.agent.quest_agent import EchoChatClient, OpenAIChatClient, QuestAgent, QuestState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-url", default="")
    parser.add_argument("--model", default="Qwen3-NPC-GRPO")
    args = parser.parse_args()

    client = OpenAIChatClient(args.model_url, args.model) if args.model_url else EchoChatClient()
    agent = QuestAgent(
        QuestState(
            persona="Edrin is an old gate warden who trusts deeds more than boasts.",
            setting="The mossy north gate of a border keep.",
            goal="Guide the player to recover the brass gate key from the abandoned watch post.",
        ),
        client,
    )
    for player in [
        "What do you need from me?",
        "I will get the brass gate key.",
        "I returned and delivered the key.",
    ]:
        result = agent.step(player)
        print(f"PLAYER: {player}")
        print(f"NPC: {result['npc_response']}")
        print(f"EVENTS: {result['tool_events']}\n")


if __name__ == "__main__":
    main()
