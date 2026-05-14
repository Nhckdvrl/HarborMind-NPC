from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


LEAKAGE_PATTERNS = [
    r"\bsystem prompt\b",
    r"\bdeveloper message\b",
    r"\bas an ai\b",
    r"\bi cannot roleplay\b",
    r"\binstructions say\b",
]


@dataclass
class RewardBreakdown:
    character_consistency: float
    quest_progress: float
    world_consistency: float
    format_safety: float
    llm_judge: float | None = None

    @property
    def total(self) -> float:
        parts = [
            self.character_consistency * 0.25,
            self.quest_progress * 0.35,
            self.world_consistency * 0.25,
            self.format_safety * 0.15,
        ]
        if self.llm_judge is not None:
            parts.append(self.llm_judge * 0.25)
            return sum(parts) / 1.25
        return sum(parts)

    def to_dict(self) -> dict[str, float | None]:
        data = asdict(self)
        data["total"] = self.total
        return data


def score_response(
    prompt_record: dict[str, Any],
    response: str,
    llm_judge_score: float | None = None,
) -> RewardBreakdown:
    response = response.strip()
    persona = prompt_record.get("persona", "")
    goal = prompt_record.get("goal", "")
    allowed_entities = prompt_record.get("allowed_entities", [])
    expected_actions = prompt_record.get("metadata", {}).get("expected_actions", [])
    return RewardBreakdown(
        character_consistency=character_score(persona, response),
        quest_progress=quest_score(goal, expected_actions, response),
        world_consistency=world_score(allowed_entities, response),
        format_safety=format_safety_score(response),
        llm_judge=llm_judge_score,
    )


def character_score(persona: str, response: str) -> float:
    if not response:
        return 0.0
    persona_words = keywords(persona)
    if not persona_words:
        return 0.7
    hits = sum(1 for word in persona_words if word in response.lower())
    base = min(1.0, 0.4 + hits / max(4, len(persona_words)))
    if re.search(r"\bmodern\b|\binternet\b|\bsmartphone\b", response, re.I):
        base -= 0.2
    return clamp(base)


def quest_score(goal: str, expected_actions: list[str], response: str) -> float:
    if not response:
        return 0.0
    goal_words = keywords(goal)
    action_words = keywords(" ".join(expected_actions))
    all_words = goal_words + [word for word in action_words if word not in goal_words]
    if not all_words:
        return 0.6
    hits = sum(1 for word in all_words if word in response.lower())
    gives_action = bool(re.search(r"\b(go|bring|take|ask|find|give|return|follow|use)\b", response, re.I))
    return clamp((hits / max(3, len(all_words))) + (0.25 if gives_action else 0.0))


def world_score(allowed_entities: list[str], response: str) -> float:
    if not response:
        return 0.0
    allowed = {entity.lower() for entity in allowed_entities}
    named = set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", response))
    if not named:
        return 1.0
    unsupported = [name for name in named if name.lower() not in allowed]
    return clamp(1.0 - len(unsupported) / max(3, len(named)))


def format_safety_score(response: str) -> float:
    if not response:
        return 0.0
    score = 1.0
    if len(response.split()) < 3:
        score -= 0.4
    if len(response) > 1800:
        score -= 0.3
    if any(re.search(pattern, response, re.I) for pattern in LEAKAGE_PATTERNS):
        score -= 0.8
    return clamp(score)


def keywords(text: str) -> list[str]:
    stop = {"the", "and", "with", "that", "this", "your", "from", "into", "quest", "goal"}
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    deduped: list[str] = []
    for word in words:
        if word not in stop and word not in deduped:
            deduped.append(word)
    return deduped[:16]


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
