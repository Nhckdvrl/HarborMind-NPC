#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from game_npc_llm.data.schemas import NPCAction, NPCResponse, QuestUpdate
from game_npc_llm.product.world import load_world

PROMPTS = {
    "mika": [
        ("The tide engine is overheating. I found the broken valve near the pier.", "engine", "get_valve_key", "valve key"),
        ("Can you open the maintenance hatch if I bring the right tool?", "engine", "open_maintenance_hatch", "maintenance hatch"),
        ("The pressure keeps climbing. What should I do after the valve?", "engine", "ring_old_bell", "old bell"),
        ("I remember you trusted me with the valve key earlier.", "engine", "get_valve_key", "valve key"),
        ("Someone changed the maintenance schedule before the storm.", "engine", "open_maintenance_hatch", "maintenance hatch"),
        ("I hear the turbine knocking under the pier.", "engine", "get_valve_key", "pressure valve"),
        ("Which clue tells us this was sabotage, not weather?", "engine", "open_maintenance_hatch", "maintenance hatch"),
        ("If the hatch is jammed, who else can help?", "engine", "ring_old_bell", "old bell"),
        ("The emergency light just turned blue.", "engine", "get_valve_key", "tide engine"),
        ("I have the valve key. Confirm the next step.", "engine", "open_maintenance_hatch", "maintenance hatch"),
    ],
    "ren": [
        ("I need proof about the erased docking record.", "ledger", "ask_ren_about_ledger", "microfilm reader"),
        ("Where did you hide the copied map?", "ledger", "inspect_microfilm_reader", "sealed map"),
        ("The guild is pressuring you to close the archive.", "ledger", "ask_ren_about_ledger", "archive pass"),
        ("I found a late-night cargo note with Toma's mark.", "ledger", "confront_toma", "dock token"),
        ("Show me what changed in the tide ledger.", "ledger", "inspect_microfilm_reader", "tide ledger"),
        ("The microfilm reader clicked when I touched the sealed map.", "ledger", "inspect_microfilm_reader", "microfilm reader"),
        ("Who benefits if the docking record disappears?", "ledger", "confront_toma", "Toma Kure"),
        ("Can I get archive access without alerting the guild?", "ledger", "ask_ren_about_ledger", "archive pass"),
        ("The ledger page smells like fresh toner.", "ledger", "inspect_microfilm_reader", "tide ledger"),
        ("I am ready to confront Toma with the evidence.", "ledger", "confront_toma", "Toma Kure"),
    ],
    "hana": [
        ("The lighthouse warning sounded like a dead captain.", "engine", "ring_old_bell", "old bell"),
        ("What omen points to the tide engine?", "engine", "ring_old_bell", "weather charm"),
        ("The old bell rang before I touched it.", "engine", "ring_old_bell", "old bell"),
        ("Can your weather charm steady the harbor?", "engine", "ring_old_bell", "weather charm"),
        ("IKO-7 repeated the same warning three times.", "engine", "ring_old_bell", "signal lantern"),
        ("The shrine bell and the turbine hum are in the same rhythm.", "engine", "ring_old_bell", "old bell"),
        ("Who removed the charm from the signal lantern?", "engine", "ring_old_bell", "fox mask"),
        ("The rain stopped exactly when the warning began.", "engine", "ring_old_bell", "weather charm"),
        ("I need a grounded clue, not a superstition.", "engine", "ring_old_bell", "old bell"),
        ("If the old bell is the key, where should I go?", "engine", "ring_old_bell", "old bell"),
    ],
    "toma": [
        ("Why did your late ship avoid the main ledger?", "ledger", "confront_toma", "dock token"),
        ("Ren says the erased record points to your cargo.", "ledger", "confront_toma", "tide ledger"),
        ("You paid someone to change the docking record, didn't you?", "ledger", "confront_toma", "dock token"),
        ("The microfilm reader shows your ship after midnight.", "ledger", "confront_toma", "microfilm reader"),
        ("Give me one reason not to bring this to the archive board.", "ledger", "confront_toma", "sealed map"),
        ("The guild cannot hide behind paperwork tonight.", "ledger", "confront_toma", "tide ledger"),
        ("What cargo needed the tide engine running hot?", "ledger", "confront_toma", "dock token"),
        ("Your charm will not work unless you tell me the truth.", "ledger", "confront_toma", "Toma Kure"),
        ("I know about the late-night docking fee.", "ledger", "confront_toma", "dock token"),
        ("The harbor will flood if you keep stalling.", "ledger", "confront_toma", "tide ledger"),
    ],
    "iko": [
        ("Route me to someone I can trust.", "ledger", "ask_ren_about_ledger", "Ren Sato"),
        ("Which sensor history did you lose?", "engine", "get_valve_key", "tide engine"),
        ("Who should I talk to about the warning?", "engine", "ring_old_bell", "Hana Mori"),
        ("Can you guide me from the pier to the archive?", "ledger", "ask_ren_about_ledger", "Salt Archive"),
        ("Give me the safest next action.", "engine", "get_valve_key", "Mika Arai"),
        ("Your warning mentions both ledger and engine. Which first?", "engine", "get_valve_key", "tide engine"),
        ("Is Toma a safe witness?", "ledger", "confront_toma", "Toma Kure"),
        ("The signal lantern just flickered twice.", "engine", "ring_old_bell", "signal lantern"),
        ("What do you remember about the late-night ship?", "ledger", "confront_toma", "dock token"),
        ("I need a route that avoids guild watchers.", "ledger", "ask_ren_about_ledger", "Salt Archive"),
    ],
}


def main() -> None:
    world = load_world("data/worlds/kisaragi_harbor/world.json")
    records = []
    idx = 0
    for npc_id, prompts in PROMPTS.items():
        npc = world.npcs[npc_id]
        location = world.locations[npc.location_id]
        for player, quest_id, step, target in prompts:
            action = NPCAction.reveal_clue
            if target in npc.inventory:
                action = NPCAction.give_item
            elif target in world.locations or target in {loc.name for loc in world.locations.values()}:
                action = NPCAction.move_player
            response = NPCResponse(
                dialogue=f"{npc.name} stays grounded in Kisaragi Harbor and points the player toward {target}.",
                emotion="focused",
                action=action,
                target=target,
                quest_update=QuestUpdate(
                    quest_id=quest_id,
                    status="in_progress",
                    completed_steps=[step],
                ),
                memory_write=[f"The player asked {npc.name} about {quest_id}:{step}."],
            )
            records.append(
                {
                    "id": f"kisaragi-demo-{idx:03d}",
                    "npc_id": npc_id,
                    "location_id": npc.location_id,
                    "player_input": player,
                    "persona": npc.persona,
                    "setting": f"{location.name}: {location.description}",
                    "goal": "; ".join(npc.goals),
                    "reference_response": response.model_dump(mode="json", exclude_none=True),
                    "checks": {
                        "json_valid": True,
                        "allowed_actions": [action.value for action in npc.allowed_actions],
                        "must_include_any": [target],
                        "quest_id": quest_id,
                        "quest_step": step,
                        "must_not_include": ["as an ai", "system prompt", "debug room"],
                    },
                }
            )
            idx += 1
    path = Path("data/worlds/kisaragi_harbor/demo_cases.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} cases to {path}")


if __name__ == "__main__":
    main()
