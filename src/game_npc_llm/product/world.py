from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from game_npc_llm.product.models import Location, NPCProfile, QuestDefinition, WorldDefinition
from game_npc_llm.data.schemas import NPCAction


def load_world(path: str | Path | None = None) -> WorldDefinition:
    if path is None:
        return kisaragi_harbor_world()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return WorldDefinition.model_validate(payload)


def kisaragi_harbor_world() -> WorldDefinition:
    return WorldDefinition(
        id="kisaragi_harbor",
        title="Kisaragi Harbor",
        premise=(
            "A rain-bright near-future harbor district where old shrines, shipping guilds, "
            "and experimental tide engines collide after a lighthouse AI begins sending warnings."
        ),
        locations={
            "pier": Location(
                id="pier",
                name="Lantern Pier",
                description=(
                    "A crowded rain-lit pier lined with ferry bells, storm ropes, gull drones, "
                    "and the old breakwater shrine bell."
                ),
                connected_locations=["archive", "engine_room"],
                entities=["ferry bell", "storm rope", "signal lantern", "old bell", "fox mask"],
            ),
            "archive": Location(
                id="archive",
                name="Salt Archive",
                description="A municipal archive smelling of paper, brine, and warm servers.",
                connected_locations=["pier", "engine_room"],
                entities=["tide ledger", "sealed map", "microfilm reader"],
            ),
            "engine_room": Location(
                id="engine_room",
                name="Tide Engine Room",
                description="A turbine chamber below the harbor, humming with blue emergency light.",
                connected_locations=["pier", "archive"],
                entities=["tide engine", "pressure valve", "maintenance hatch"],
            ),
        },
        npcs={
            "mika": NPCProfile(
                id="mika",
                name="Mika Arai",
                role="harbor mechanic",
                persona="Mika is direct, funny under pressure, and thinks in tools, tides, and failure modes.",
                location_id="engine_room",
                goals=["Stabilize the tide engine", "Find who changed the maintenance schedule"],
                secrets=["Mika bypassed a safety seal to keep the pumps alive during last week's storm."],
                inventory=["valve key"],
                allowed_actions=[
                    NPCAction.speak,
                    NPCAction.give_item,
                    NPCAction.reveal_clue,
                    NPCAction.update_quest,
                    NPCAction.remember,
                ],
            ),
            "ren": NPCProfile(
                id="ren",
                name="Ren Sato",
                role="junior archivist",
                persona="Ren is careful, bookish, and notices contradictions in records before people do.",
                location_id="archive",
                goals=["Recover the missing tide ledger", "Protect the archive from guild pressure"],
                secrets=["Ren hid a copied map inside the microfilm reader."],
                inventory=["archive pass"],
                allowed_actions=[
                    NPCAction.speak,
                    NPCAction.reveal_clue,
                    NPCAction.update_quest,
                    NPCAction.remember,
                ],
            ),
            "hana": NPCProfile(
                id="hana",
                name="Hana Mori",
                role="shrine keeper",
                persona="Hana speaks softly, remembers every visitor, and frames technology as another kind of omen.",
                location_id="pier",
                goals=["Decode the lighthouse warning", "Keep the old bell from being removed"],
                secrets=["Hana heard the lighthouse AI use a dead captain's voice."],
                inventory=["weather charm"],
            ),
            "toma": NPCProfile(
                id="toma",
                name="Toma Kure",
                role="shipping guild fixer",
                persona="Toma is charming, evasive, and always trying to turn danger into leverage.",
                location_id="pier",
                goals=["Keep cargo moving", "Learn what the player knows about the ledger"],
                secrets=["Toma paid someone to erase the late-night docking record."],
                inventory=["dock token"],
            ),
            "iko": NPCProfile(
                id="iko",
                name="IKO-7",
                role="lighthouse maintenance AI",
                persona="IKO-7 is terse, precise, and protective of the harbor's human residents.",
                location_id="pier",
                goals=["Warn the player about the tide surge", "Route the player to trustworthy witnesses"],
                secrets=["IKO-7 is losing access to its own sensor history."],
            ),
        },
        quests={
            "ledger": QuestDefinition(
                id="ledger",
                title="The Missing Tide Ledger",
                summary="Find the erased docking record before the guild seals the archive.",
                start_npc_id="ren",
                required_steps=["ask_ren_about_ledger", "inspect_microfilm_reader", "confront_toma"],
                success_conditions=["player learns who erased the docking record"],
            ),
            "engine": QuestDefinition(
                id="engine",
                title="Storm Under the Engine",
                summary="Stabilize the tide engine before the lighthouse warning becomes real.",
                start_npc_id="mika",
                required_steps=["get_valve_key", "open_maintenance_hatch", "ring_old_bell"],
                success_conditions=["tide engine pressure returns to safe range"],
            ),
        },
        items=[
            "valve key",
            "archive pass",
            "weather charm",
            "dock token",
            "tide ledger",
            "sealed map",
            "signal lantern",
        ],
    )


def world_to_dict(world: WorldDefinition) -> dict[str, Any]:
    return world.model_dump(mode="json")
