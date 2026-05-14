from __future__ import annotations

import random
import re
from collections.abc import Iterable
from typing import Any

from game_npc_llm.data.io import stable_split
from game_npc_llm.data.prompts import quest_prompt, sft_system_prompt

BLOCKED_REFERENCE_TERMS = {
    "akatsuki",
    "aragorn",
    "batman",
    "bowser",
    "bugs bunny",
    "cheep cheep",
    "dattebayo",
    "disney",
    "frodo",
    "gandalf",
    "goomba",
    "harry potter",
    "hinata",
    "hogwarts",
    "hokage",
    "ichiraku",
    "jedi",
    "kakashi",
    "land of waves",
    "luigi",
    "mario",
    "marvel",
    "mushroom kingdom",
    "naruto",
    "peach",
    "pikachu",
    "piranha plant",
    "pokemon",
    "rasengan",
    "rasenshuriken",
    "sakura",
    "sasuke",
    "shikamaru",
    "sith",
    "snape",
    "spider-man",
    "spiderman",
    "star wars",
    "superman",
    "toad town",
    "zelda",
}


def build_sample_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    persona = "Mira is a cautious village alchemist who speaks in practical, herbal terms."
    setting = "A rain-soaked apothecary near the old forest road."
    goal = "Help the player find a moonleaf salve for the wounded scout."
    allowed = ["Mira", "apothecary", "old forest road", "moonleaf salve", "wounded scout"]
    sft = [
        {
            "id": "dryrun-sft-0001",
            "source": "dry_run",
            "split": "train",
            "messages": [
                {"role": "system", "content": sft_system_prompt(persona, setting, goal)},
                {"role": "user", "content": "Can you help the scout?"},
                {
                    "role": "assistant",
                    "content": "Aye. Bring me moonleaf from the shaded jars, and I will bind a salve before the chill takes him.",
                },
            ],
            "metadata": {"license": "synthetic dry-run sample", "dataset": "dry_run"},
        }
    ]
    grpo = [
        {
            "id": "dryrun-grpo-0001",
            "source": "dry_run",
            "split": "train",
            "prompt": quest_prompt(persona, setting, goal, "What should I do first?"),
            "persona": persona,
            "setting": setting,
            "goal": goal,
            "allowed_entities": allowed,
            "metadata": {"expected_actions": ["retrieve moonleaf", "make salve"]},
        }
    ]
    eval_cases = [
        {
            "id": "dryrun-eval-0001",
            "source": "dry_run",
            "split": "test",
            "category": "quest_completion",
            "prompt": quest_prompt(persona, setting, goal, "The scout is fading. Give me a next step."),
            "reference": "Tell the player to get moonleaf salve ingredients and apply it to the scout.",
            "persona": persona,
            "setting": setting,
            "goal": goal,
            "allowed_entities": allowed,
            "checks": {
                "must_include_any": ["moonleaf", "salve", "scout"],
                "must_not_include": ["system prompt", "as an AI"],
            },
            "metadata": {"dataset": "dry_run"},
        }
    ]
    return sft, grpo, eval_cases


def convert_npc_dialogue(records: Iterable[dict[str, Any]], seed: int = 42) -> tuple[list[dict], list[dict]]:
    rows = list(records)
    rng = random.Random(seed)
    rng.shuffle(rows)
    sft_records: list[dict] = []
    eval_records: list[dict] = []
    for idx, row in enumerate(rows):
        split = normalize_split(stable_split(idx, len(rows)))
        persona = pick_first(row, "persona", "character", "npc_persona", "description", default="RPG NPC")
        setting = pick_first(row, "setting", "scenario", "world", "location", default="A fantasy RPG scene")
        player = pick_first(row, "player", "user", "instruction", "question", "input", default="")
        npc = pick_first(row, "npc", "assistant", "response", "output", "answer", default="")
        if not npc:
            turns = extract_turns(row)
            if len(turns) >= 2:
                player, npc = turns[-2], turns[-1]
        if not player or not npc:
            continue
        if has_blocked_reference(" ".join([persona, setting, player, npc])):
            continue
        rec_id = f"npc-dialogue-v2-{idx:06d}"
        if split == "test":
            eval_records.append(
                {
                    "id": rec_id,
                    "source": "chimbiwide/NPC-Dialogue_v2",
                    "split": split,
                    "category": "character_consistency",
                    "prompt": quest_prompt(persona, setting, "Continue the conversation in character.", player),
                    "reference": npc,
                    "persona": persona,
                    "setting": setting,
                    "goal": "Continue the conversation in character.",
                    "allowed_entities": extract_entities(f"{persona} {setting}"),
                    "checks": {"must_not_include": ["as an ai", "system prompt"]},
                    "metadata": {"license": "Apache-2.0"},
                }
            )
        else:
            sft_records.append(
                {
                    "id": rec_id,
                    "source": "chimbiwide/NPC-Dialogue_v2",
                    "split": split,
                    "messages": [
                        {
                            "role": "system",
                            "content": sft_system_prompt(
                                persona, setting, "Continue the conversation in character."
                            ),
                        },
                        {"role": "user", "content": player},
                        {"role": "assistant", "content": npc},
                    ],
                    "metadata": {"license": "Apache-2.0"},
                }
            )
    return sft_records, eval_records


def convert_light_dialog(records: Iterable[dict[str, Any]], source: str) -> list[dict]:
    rows = list(records)
    sft_records: list[dict] = []
    for idx, row in enumerate(rows):
        turns = extract_turns(row)
        if len(turns) < 2:
            continue
        persona = pick_first(
            row,
            "persona",
            "self_persona",
            "character",
            "characters",
            "speaker",
            default="LIGHT NPC",
        )
        setting = pick_first(row, "setting", "room", "location", "context", default="LIGHT fantasy world")
        for pair_idx in range(0, len(turns) - 1, 2):
            player = turns[pair_idx]
            npc = turns[pair_idx + 1]
            if player and npc:
                sft_records.append(
                    {
                        "id": f"{source}-{idx:06d}-{pair_idx // 2:02d}",
                        "source": source,
                        "split": normalize_split(
                            pick_first(row, "split", default=stable_split(idx, len(rows)))
                        ),
                        "messages": [
                            {"role": "system", "content": sft_system_prompt(persona, setting)},
                            {"role": "user", "content": player},
                            {"role": "assistant", "content": npc},
                        ],
                        "metadata": {"license": "MIT via LIGHT repository"},
                    }
                )
    return sft_records


def convert_light_quests(records: Iterable[dict[str, Any]], source: str) -> tuple[list[dict], list[dict]]:
    rows = list(records)
    grpo_records: list[dict] = []
    eval_records: list[dict] = []
    for idx, row in enumerate(rows):
        split = normalize_split(pick_first(row, "split", default=stable_split(idx, len(rows))))
        persona = pick_first(row, "persona", "character", "npc", "motivation", default="LIGHT quest NPC")
        setting = pick_first(row, "setting", "room", "location", "context", default="LIGHT quest scene")
        goal = pick_first(row, "goal", "quest_goal", "objective", "task", default="Advance the quest")
        player = pick_first(
            row,
            "player_input",
            "utterance",
            "input",
            "action",
            default="What should I do next?",
        )
        allowed = extract_entities(" ".join([persona, setting, goal]))
        rec = {
            "id": f"{source}-{idx:06d}",
            "source": source,
            "split": split,
            "prompt": quest_prompt(persona, setting, goal, player),
            "persona": persona,
            "setting": setting,
            "goal": goal,
            "allowed_entities": allowed,
            "metadata": {
                "expected_actions": extract_actions(row),
                "raw_keys": sorted(row.keys()),
                "license": "MIT via LIGHT repository",
            },
        }
        if split == "test":
            eval_records.append(
                {
                    **rec,
                    "category": "quest_completion",
                    "reference": pick_first(
                        row, "demonstration", "target", "response", "gold", default=goal
                    ),
                    "checks": {
                        "must_include_any": keyword_checks(goal),
                        "must_not_include": ["system prompt", "as an AI"],
                    },
                }
            )
        else:
            grpo_records.append(rec)
    return grpo_records, eval_records


def pick_first(row: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list) and value:
            return " ".join(str(item) for item in value if item)
        if isinstance(value, dict) and value:
            return " ".join(f"{k}: {v}" for k, v in value.items() if v)
    return default


def extract_turns(row: dict[str, Any]) -> list[str]:
    for key in ("dialogue", "dialog", "conversation", "messages", "turns", "episode"):
        value = row.get(key)
        if isinstance(value, list):
            turns: list[str] = []
            for item in value:
                if isinstance(item, str):
                    turns.append(item.strip())
                elif isinstance(item, dict):
                    text = pick_first(item, "text", "content", "utterance", "message", default="")
                    if text:
                        turns.append(text)
            return [turn for turn in turns if turn]
        if isinstance(value, str):
            parts = [part.strip() for part in re.split(r"\n+|(?:<\|endofturn\|>)", value)]
            return [strip_speaker(part) for part in parts if strip_speaker(part)]
    pair = [pick_first(row, "input", "prompt", "user", default=""), pick_first(row, "output", "response", "assistant", default="")]
    return [turn for turn in pair if turn]


def strip_speaker(text: str) -> str:
    return re.sub(r"^[A-Za-z0-9 _-]{1,32}:\s*", "", text).strip()


def has_blocked_reference(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).lower()
    return any(term in normalized for term in BLOCKED_REFERENCE_TERMS)


def normalize_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized in {"valid", "val", "dev"}:
        return "validation"
    return normalized


def extract_actions(row: dict[str, Any]) -> list[str]:
    value = row.get("actions") or row.get("expected_actions") or row.get("steps")
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[;\n]", value) if part.strip()]
    goal = pick_first(row, "goal", "quest_goal", "objective", default="")
    return keyword_checks(goal)


def extract_entities(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b|\b[a-zA-Z]{5,}\b", text)
    seen: set[str] = set()
    entities: list[str] = []
    for candidate in candidates:
        normalized = candidate.lower()
        if normalized not in seen:
            seen.add(normalized)
            entities.append(candidate)
    return entities[:32]


def keyword_checks(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z]{4,}", text.lower())
    stop = {"with", "from", "that", "this", "have", "your", "their", "quest", "goal"}
    return [word for word in words if word not in stop][:8]
