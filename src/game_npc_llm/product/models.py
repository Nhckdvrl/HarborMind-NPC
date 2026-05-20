from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from game_npc_llm.data.schemas import NPCAction, NPCResponse


class NPCProfile(BaseModel):
    id: str
    name: str
    role: str
    persona: str
    location_id: str
    goals: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)
    inventory: list[str] = Field(default_factory=list)
    allowed_actions: list[NPCAction] = Field(
        default_factory=lambda: [NPCAction.speak, NPCAction.reveal_clue, NPCAction.remember]
    )


class Location(BaseModel):
    id: str
    name: str
    description: str
    connected_locations: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)


class QuestDefinition(BaseModel):
    id: str
    title: str
    summary: str
    start_npc_id: str
    required_steps: list[str]
    success_conditions: list[str]


class WorldDefinition(BaseModel):
    id: str
    title: str
    premise: str
    locations: dict[str, Location]
    npcs: dict[str, NPCProfile]
    quests: dict[str, QuestDefinition]
    items: list[str] = Field(default_factory=list)

    @field_validator("locations", "npcs", "quests")
    @classmethod
    def _non_empty_mapping(cls, value: dict) -> dict:
        if not value:
            raise ValueError("world mappings must be non-empty")
        return value

    def allowed_entities(self) -> set[str]:
        entities = set(self.items)
        entities.update(location.name for location in self.locations.values())
        entities.update(self.locations)
        entities.update(npc.name for npc in self.npcs.values())
        entities.update(self.npcs)
        for location in self.locations.values():
            entities.update(location.entities)
        return {entity.lower() for entity in entities if entity}


class GameState(BaseModel):
    session_id: str = "demo"
    current_location_id: str = "pier"
    player_profile: dict[str, str] = Field(default_factory=dict)
    inventory: list[str] = Field(default_factory=list)
    quest_status: dict[str, Literal["not_started", "in_progress", "completed", "failed"]] = (
        Field(default_factory=dict)
    )
    completed_steps: dict[str, list[str]] = Field(default_factory=dict)
    recent_turns: list[str] = Field(default_factory=list)

    def remember_turn(self, speaker: str, text: str, limit: int = 12) -> None:
        clean = text.strip()
        if clean:
            self.recent_turns.append(f"{speaker}: {clean}")
            self.recent_turns = self.recent_turns[-limit:]


class ChatRequest(BaseModel):
    npc_id: str
    player_input: str
    session_id: str = "demo"


class ChatResult(BaseModel):
    npc_id: str
    response: NPCResponse
    state: GameState
    memory_hits: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
