from __future__ import annotations

import json
import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

SFT_REQUIRED_KEYS = {"id", "source", "split", "messages", "metadata"}
PREFERENCE_REQUIRED_KEYS = {"id", "source", "split", "prompt", "chosen", "rejected", "metadata"}
EVAL_REQUIRED_KEYS = {
    "id",
    "source",
    "split",
    "category",
    "prompt",
    "reference",
    "persona",
    "setting",
    "goal",
    "allowed_entities",
    "checks",
    "metadata",
}


class NPCAction(str, Enum):
    speak = "speak"
    give_item = "give_item"
    request_item = "request_item"
    reveal_clue = "reveal_clue"
    update_quest = "update_quest"
    move_player = "move_player"
    refuse = "refuse"
    remember = "remember"
    wait = "wait"


class QuestUpdate(BaseModel):
    quest_id: str
    status: Literal["not_started", "in_progress", "completed", "failed"]
    completed_steps: list[str] = Field(default_factory=list)


class NPCResponse(BaseModel):
    dialogue: str = Field(min_length=1, max_length=1200)
    emotion: str = Field(default="neutral", min_length=1, max_length=64)
    action: NPCAction = NPCAction.speak
    target: str | None = None
    quest_update: QuestUpdate | None = None
    memory_write: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)

    @field_validator("memory_write", "safety_flags")
    @classmethod
    def _strip_lists(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    def to_json_text(self) -> str:
        return self.model_dump_json(exclude_none=True)


NPC_RESPONSE_JSON_SCHEMA = NPCResponse.model_json_schema()


def validate_record(record: dict, schema: str) -> list[str]:
    if schema == "sft":
        return _validate_sft(record)
    if schema == "preference":
        return _validate_preference(record)
    if schema == "eval":
        return _validate_eval(record)
    raise ValueError(f"Unknown schema: {schema}")


def _missing(record: dict, required: set[str]) -> list[str]:
    return [f"missing key: {key}" for key in sorted(required - set(record))]


def _validate_sft(record: dict) -> list[str]:
    errors = _missing(record, SFT_REQUIRED_KEYS)
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("messages must contain at least user and assistant turns")
    else:
        for idx, message in enumerate(messages):
            if message.get("role") not in {"system", "user", "assistant"}:
                errors.append(f"messages[{idx}].role is invalid")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                errors.append(f"messages[{idx}].content must be non-empty")
    return errors


def _validate_preference(record: dict) -> list[str]:
    errors = _missing(record, PREFERENCE_REQUIRED_KEYS)
    for key in ("prompt", "chosen", "rejected"):
        if not isinstance(record.get(key), str) or not record.get(key, "").strip():
            errors.append(f"{key} must be a non-empty string")
    if record.get("chosen") == record.get("rejected"):
        errors.append("chosen and rejected must differ")
    return errors


def _validate_eval(record: dict) -> list[str]:
    errors = _missing(record, EVAL_REQUIRED_KEYS)
    for key in ("prompt", "persona", "setting", "reference"):
        if not isinstance(record.get(key), str):
            errors.append(f"{key} must be a string")
    if not isinstance(record.get("checks"), dict):
        errors.append("checks must be an object")
    return errors


def parse_npc_response(text: str) -> tuple[NPCResponse | None, list[str]]:
    candidate = extract_json_object(text)
    if not candidate:
        return None, ["missing JSON object"]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON: {exc}"]
    try:
        return NPCResponse.model_validate(payload), []
    except ValidationError as exc:
        return None, [error["msg"] for error in exc.errors()]


def repair_npc_response(text: str, fallback_dialogue: str | None = None) -> NPCResponse:
    parsed, _ = parse_npc_response(text)
    if parsed is not None:
        return parsed
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    dialogue = fallback_dialogue or cleaned or "I need a moment to gather my thoughts."
    return NPCResponse(
        dialogue=dialogue[:1200],
        emotion="uncertain",
        action=NPCAction.speak,
        safety_flags=["repaired_output"],
    )


def extract_json_object(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if match:
        return match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return None
