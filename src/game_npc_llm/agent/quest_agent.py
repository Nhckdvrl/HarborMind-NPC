from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from game_npc_llm.data.prompts import sft_system_prompt


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


@dataclass
class QuestState:
    persona: str
    setting: str
    goal: str
    inventory: list[str] = field(default_factory=list)
    memory: list[str] = field(default_factory=list)
    quest_status: str = "not_started"
    completed_steps: list[str] = field(default_factory=list)


class OpenAIChatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY") -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=256,
        )
        return response.choices[0].message.content or ""


class QuestAgent:
    def __init__(self, state: QuestState, chat_client: ChatClient) -> None:
        self.state = state
        self.chat_client = chat_client

    def step(self, player_input: str) -> dict[str, object]:
        tool_events = self._advance_state(player_input)
        messages = self._messages(player_input, tool_events)
        npc_response = self.chat_client.complete(messages).strip()
        self.state.memory.append(f"Player: {player_input}")
        self.state.memory.append(f"NPC: {npc_response}")
        return {
            "npc_response": npc_response,
            "state": self.state,
            "tool_events": tool_events,
        }

    def _messages(self, player_input: str, tool_events: list[str]) -> list[dict[str, str]]:
        system = sft_system_prompt(self.state.persona, self.state.setting, self.state.goal)
        state_text = (
            f"Inventory: {', '.join(self.state.inventory) or 'empty'}\n"
            f"Quest status: {self.state.quest_status}\n"
            f"Completed steps: {', '.join(self.state.completed_steps) or 'none'}\n"
            f"Recent memory: {' | '.join(self.state.memory[-6:]) or 'none'}\n"
            f"State updates this turn: {' | '.join(tool_events) or 'none'}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{state_text}\nPlayer says: {player_input}"},
        ]

    def _advance_state(self, player_input: str) -> list[str]:
        text = player_input.lower()
        events: list[str] = []
        if self.state.quest_status == "not_started":
            self.state.quest_status = "in_progress"
            events.append("quest_started")
        for item in re.findall(r"\b(?:take|get|pick up|bring)\s+([a-zA-Z][a-zA-Z -]{2,32})", text):
            clean = item.strip(" .,!?:;").lower()
            if clean and clean not in self.state.inventory:
                self.state.inventory.append(clean)
                events.append(f"inventory_added:{clean}")
        goal_words = {word for word in re.findall(r"[a-z]{4,}", self.state.goal.lower())}
        if goal_words and any(word in text for word in goal_words):
            if "goal_discussed" not in self.state.completed_steps:
                self.state.completed_steps.append("goal_discussed")
                events.append("goal_discussed")
        if re.search(r"\b(done|completed|finished|gave|delivered|returned)\b", text):
            self.state.quest_status = "completed"
            events.append("quest_completed")
        return events


class EchoChatClient:
    def complete(self, messages: list[dict[str, str]]) -> str:
        user = messages[-1]["content"]
        if "Quest status: completed" in user:
            return "Then our work is done. Keep your wits sharp on the road ahead."
        return "Stay close to the task: gather what the quest needs, then return to me for the next step."
