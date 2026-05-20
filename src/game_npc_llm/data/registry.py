from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_path: str
    license: str
    role: str
    default_limit: int
    trainable_public: bool = True
    config: str | None = None


DATASETS = {
    "roleplay_npc_quest": DatasetSpec(
        name="roleplay_npc_quest",
        hf_path="chimbiwide/RolePlay-NPC-Quest",
        license="Apache-2.0",
        role="primary_sft",
        default_limit=20500,
    ),
    "npc_dialogue_v2": DatasetSpec(
        name="npc_dialogue_v2",
        hf_path="chimbiwide/NPC-Dialogue_v2",
        config="dialogue",
        license="Apache-2.0",
        role="auxiliary_sft",
        default_limit=3500,
    ),
    "soda": DatasetSpec(
        name="soda",
        hf_path="allenai/soda",
        license="CC-BY-4.0",
        role="social_emotion_sft",
        default_limit=10000,
    ),
    "openhermes_roleplay_preferences": DatasetSpec(
        name="openhermes_roleplay_preferences",
        hf_path="vicgalle/OpenHermesPreferences-roleplay",
        license="other",
        role="optional_preference",
        default_limit=3000,
        trainable_public=False,
    ),
}


BLOCKED_REFERENCE_TERMS = {
    "akatsuki",
    "akane",
    "akeno",
    "alucard",
    "aragorn",
    "austin powers",
    "batman",
    "beacon hills",
    "bowser",
    "codsworth",
    "disney",
    "fallout",
    "fate/grand",
    "fazbear",
    "freddy",
    "frodo",
    "gandalf",
    "ganyu",
    "gridman",
    "harry potter",
    "hogwarts",
    "hololive",
    "ina'nis",
    "jedi",
    "mario",
    "mashu",
    "marvel",
    "mumei",
    "naruto",
    "pikachu",
    "pokemon",
    "krueger",
    "kronii",
    "laplus",
    "rias",
    "scp-",
    "sith",
    "spider-man",
    "star wars",
    "superman",
    "toriel",
    "zelda",
}
