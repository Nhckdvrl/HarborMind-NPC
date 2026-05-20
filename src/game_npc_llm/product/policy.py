from __future__ import annotations

import json
import os
from typing import Protocol

from game_npc_llm.data.schemas import NPCAction, NPCResponse, QuestUpdate


class PolicyClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class OpenAICompatiblePolicyClient:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY") -> None:
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(os.getenv("NPC_MODEL_TIMEOUT", "120")),
        )
        self.model = model
        self.temperature = float(os.getenv("NPC_MODEL_TEMPERATURE", "0.2"))
        self.max_tokens = int(os.getenv("NPC_MODEL_MAX_TOKENS", "128"))

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class RulePolicyClient:
    """Deterministic local policy for tests and demos without a model server."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        user = messages[-1]["content"].lower()
        if "valve" in user or "engine" in user:
            response = NPCResponse(
                dialogue=(
                    "Take the valve key and meet me by the maintenance hatch. "
                    "If the pressure climbs again, ring the old bell at the shrine."
                ),
                emotion="urgent",
                action=NPCAction.give_item,
                target="valve key",
                quest_update=QuestUpdate(
                    quest_id="engine",
                    status="in_progress",
                    completed_steps=["get_valve_key"],
                ),
                memory_write=["The player was trusted with the valve key."],
            )
        elif "ledger" in user or "archive" in user or "record" in user:
            response = NPCResponse(
                dialogue=(
                    "The tide ledger vanished after Toma's late ship docked. "
                    "Check the microfilm reader before the guild locks the archive."
                ),
                emotion="worried",
                action=NPCAction.reveal_clue,
                target="microfilm reader",
                quest_update=QuestUpdate(
                    quest_id="ledger",
                    status="in_progress",
                    completed_steps=["ask_ren_about_ledger"],
                ),
                memory_write=["The player learned that the missing ledger points to Toma."],
            )
        else:
            response = NPCResponse(
                dialogue="Stay close to the harbor clues. Ask about the ledger or the tide engine first.",
                emotion="focused",
                action=NPCAction.speak,
                memory_write=["The player asked for general guidance."],
            )
        return json.dumps(response.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
