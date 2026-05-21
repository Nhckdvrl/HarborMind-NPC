from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from game_npc_llm.data.prompts import npc_agent_system_prompt, npc_turn_user_message
from game_npc_llm.data.schemas import NPCAction, NPCResponse, repair_npc_response
from game_npc_llm.product.memory import InMemoryMemoryStore, MemoryStore, create_memory_store
from game_npc_llm.product.models import ChatResult, GameState, WorldDefinition
from game_npc_llm.product.policy import PolicyClient, RulePolicyClient, create_policy_from_env
from game_npc_llm.product.world import load_world

# Maximum number of (user + assistant) messages kept in per-session history.
# 20 messages = 10 full turns, warm context without blowing the context window.
_MAX_HISTORY = 20


@dataclass
class ExecutionResult:
    events: list[str] = field(default_factory=list)
    visible_events: list[str] = field(default_factory=list)
    quest_delta: dict[str, list[str] | str] = field(default_factory=dict)
    inventory_delta: list[str] = field(default_factory=list)
    relationship_delta: dict[str, int] = field(default_factory=dict)


@dataclass
class GameAgent:
    world: WorldDefinition
    policy: PolicyClient
    memory: MemoryStore
    states: dict[str, GameState]
    state_path: Path | None = None

    def __post_init__(self) -> None:
        if self.state_path and self.state_path.exists():
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.states.update(
                {sid: GameState.model_validate(data) for sid, data in raw.items()}
            )

    @classmethod
    def demo(cls) -> "GameAgent":
        """Factory that auto-selects policy from env: LLM if NPC_MODEL_URL is set, else rules."""
        return cls(
            world=load_world(),
            policy=create_policy_from_env(),
            memory=create_memory_store(),
            states={},
        )

    def _persist(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {sid: state.model_dump(mode="json") for sid, state in self.states.items()}
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def state_for(self, session_id: str) -> GameState:
        if session_id not in self.states:
            self.states[session_id] = GameState(
                session_id=session_id,
                quest_status={quest_id: "not_started" for quest_id in self.world.quests},
                quest_steps={quest_id: [] for quest_id in self.world.quests},
                relationships={npc_id: 0 for npc_id in self.world.npcs},
            )
        return self.states[session_id]

    def reset(self, session_id: str = "demo") -> GameState:
        state = GameState(
            session_id=session_id,
            quest_status={quest_id: "not_started" for quest_id in self.world.quests},
            quest_steps={quest_id: [] for quest_id in self.world.quests},
            relationships={npc_id: 0 for npc_id in self.world.npcs},
        )
        self.states[session_id] = state
        self._persist()
        return state

    # ------------------------------------------------------------------
    # Core agent loop  (nanobot-inspired: observe → think → act)
    # ------------------------------------------------------------------

    def chat(self, npc_id: str, player_input: str, session_id: str = "demo") -> ChatResult:
        if npc_id not in self.world.npcs:
            raise ValueError(f"Unknown npc_id: {npc_id}")

        state = self.state_for(session_id)
        npc = self.world.npcs[npc_id]
        location = self.world.locations[npc.location_id]

        # OBSERVE: extract facts from player input, retrieve relevant memories
        extracted_memories = self._extract_player_facts(state, npc_id, player_input)
        memory_hits = self.memory.search(session_id, npc_id, player_input)

        # THINK: send full context to LLM (or rule fallback)
        if isinstance(self.policy, RulePolicyClient):
            raw = self.policy.complete_turn(self.world, state, npc_id, player_input, memory_hits)
        else:
            messages = self._build_llm_messages(npc, location, npc_id, player_input, state, memory_hits)
            raw = self.policy.complete(messages)

        response = repair_npc_response(raw)

        # ACT: execute the chosen action, update game state
        result = self._execute_response(state, npc_id, player_input, response)

        # Update dialogue history as proper chat messages (clean text, no state noise).
        # This is what gets injected into next turn's LLM context.
        state.message_history.append({"role": "user", "content": player_input})
        state.message_history.append({"role": "assistant", "content": response.dialogue})
        if len(state.message_history) > _MAX_HISTORY:
            state.message_history = state.message_history[-_MAX_HISTORY:]

        state.remember_turn("Player", player_input)
        state.remember_turn(npc.name, response.dialogue)

        for memory in extracted_memories:
            self.memory.add(session_id, npc_id, memory)
        for memory in response.memory_write:
            self.memory.add(session_id, npc_id, memory)

        self._persist()
        return ChatResult(
            npc_id=npc_id,
            response=response,
            state=state,
            memory_hits=memory_hits,
            events=result.events,
            visible_events=result.visible_events,
            quest_delta=result.quest_delta,
            inventory_delta=result.inventory_delta,
            relationship_delta=result.relationship_delta,
            next_suggestions=self._next_suggestions(state, npc_id),
        )

    def _build_llm_messages(self, npc, location, npc_id: str, player_input: str, state: GameState, memory_hits: list[str]) -> list[dict]:
        """Bundle system prompt + dialogue history + current observation into one messages list.

        This is the nanobot 'observe' step: the LLM receives its full identity, all prior
        turns as proper chat messages, and the current game state snapshot as a single context.
        """
        system = npc_agent_system_prompt(npc, self.world, location)
        user_msg = npc_turn_user_message(player_input, state, npc_id, memory_hits)
        return [
            {"role": "system", "content": system},
            *state.message_history[-_MAX_HISTORY:],
            {"role": "user", "content": user_msg},
        ]

    # ------------------------------------------------------------------
    # Game state helpers
    # ------------------------------------------------------------------

    def _extract_player_facts(self, state: GameState, npc_id: str, player_input: str) -> list[str]:
        del npc_id
        facts: list[str] = []
        name = _extract_player_name(player_input)
        if name:
            state.player_profile["name"] = name
            facts.append(f"The player's name is {name}. Do not ask for their name again.")
        return facts

    def _execute_response(
        self, state: GameState, npc_id: str, player_input: str, response: NPCResponse
    ) -> ExecutionResult:
        npc = self.world.npcs[npc_id]
        result = ExecutionResult()
        relationship_gain = self._relationship_gain(player_input, response)
        if relationship_gain:
            state.relationships[npc_id] = state.relationships.get(npc_id, 0) + relationship_gain
            result.relationship_delta[npc_id] = relationship_gain
            result.events.append(f"relationship:{npc_id}:{relationship_gain:+d}")
        always_allowed = {NPCAction.speak, NPCAction.refuse}
        if response.action not in npc.allowed_actions and response.action not in always_allowed:
            response.safety_flags.append("illegal_action_blocked")
            response.action = NPCAction.speak
            response.target = None
            result.events.append("illegal_action_blocked")
            result.visible_events.append(f"{npc.name} tries an unavailable action, so the game layer blocks it.")
        if response.action == NPCAction.refuse:
            result.visible_events.append(f"{npc.name} refuses the request and keeps the scene within safe game rules.")
        if response.action == NPCAction.give_item and response.target:
            item = response.target.lower()
            if item in {owned.lower() for owned in npc.inventory} or item in self.world.allowed_entities():
                if item not in state.inventory:
                    state.inventory.append(item)
                    result.inventory_delta.append(item)
                    result.events.append(f"inventory_added:{item}")
                    result.visible_events.append(f"Inventory updated: {item}.")
            else:
                response.safety_flags.append("unknown_item_blocked")
                result.events.append("unknown_item_blocked")
        if response.action == NPCAction.move_player and response.target:
            target = response.target.lower().replace(" ", "_")
            location_by_name = {
                loc.name.lower(): loc_id for loc_id, loc in self.world.locations.items()
            }
            target_id = location_by_name.get(response.target.lower(), target)
            current = self.world.locations[state.current_location_id]
            if target_id in current.connected_locations:
                state.current_location_id = target_id
                result.events.append(f"moved:{target_id}")
                result.visible_events.append(f"Moved to {self.world.locations[target_id].name}.")
            else:
                response.safety_flags.append("illegal_move_blocked")
                result.events.append("illegal_move_blocked")
        if response.quest_update:
            quest_id = response.quest_update.quest_id
            if quest_id in self.world.quests:
                state.quest_status[quest_id] = response.quest_update.status
                known = state.completed_steps.setdefault(quest_id, [])
                step_state = state.quest_steps.setdefault(quest_id, [])
                for step in response.quest_update.completed_steps:
                    if step in self.world.quests[quest_id].required_steps and step not in known:
                        known.append(step)
                        step_state.append(step)
                        result.events.append(f"quest_step:{quest_id}:{step}")
                        result.visible_events.append(f"Quest updated: {self.world.quests[quest_id].title} -> {step}.")
                        result.quest_delta.setdefault(quest_id, []).append(step)  # type: ignore[union-attr]
                result.quest_delta[f"{quest_id}_status"] = response.quest_update.status
                if _quest_complete(self.world.quests[quest_id].required_steps, known):
                    state.quest_status[quest_id] = "completed"
                    result.quest_delta[f"{quest_id}_status"] = "completed"
                    result.visible_events.append(f"Quest complete: {self.world.quests[quest_id].title}.")
            else:
                response.safety_flags.append("unknown_quest_blocked")
                result.events.append("unknown_quest_blocked")
        if response.action == NPCAction.reveal_clue and response.target:
            clue = response.target.strip()
            if clue and clue.lower() in self.world.allowed_entities() and clue not in state.known_clues:
                state.known_clues.append(clue)
                result.events.append(f"clue_added:{clue.lower()}")
                result.visible_events.append(f"New clue logged: {clue}.")
        self._derive_world_flags(state)
        return result

    def _relationship_gain(self, player_input: str, response: NPCResponse) -> int:
        text = player_input.lower()
        if response.action in {NPCAction.give_item, NPCAction.reveal_clue, NPCAction.update_quest}:
            return 2
        if any(term in text for term in ["please", "thanks", "thank", "help", "trust", "谢", "麻烦"]):
            return 1
        if response.action == NPCAction.refuse:
            return -1
        return 0

    def _derive_world_flags(self, state: GameState) -> None:
        clues = {clue.lower() for clue in state.known_clues}
        inventory = {item.lower() for item in state.inventory}
        state.world_flags["engine_access"] = "valve key" in inventory
        state.world_flags["ledger_evidence"] = "microfilm reader" in clues or "dock token" in clues
        state.world_flags["lighthouse_warning"] = "old bell" in clues

    def _next_suggestions(self, state: GameState, npc_id: str) -> list[str]:
        if state.quest_status.get("engine") != "completed":
            engine_steps = set(state.quest_steps.get("engine", []))
            if "get_valve_key" not in engine_steps:
                return ["Ask Mika about the tide engine.", "Ask Hana about the old bell."]
            if "open_maintenance_hatch" not in engine_steps:
                return ["Open the maintenance hatch.", "Ask Mika what the valve key unlocks."]
            return ["Ask Hana to ring the old bell.", "Ask IKO-7 about the tide surge."]
        if state.quest_status.get("ledger") != "completed":
            ledger_steps = set(state.quest_steps.get("ledger", []))
            if "ask_ren_about_ledger" not in ledger_steps:
                return ["Ask Ren for proof in the archive.", "Ask IKO-7 who is trustworthy."]
            if "inspect_microfilm_reader" not in ledger_steps:
                return ["Inspect the microfilm reader.", "Ask Ren about the copied map."]
            return ["Confront Toma with the microfilm clue.", "Ask Ren about the copied map."]
        return [f"Follow up with {self.world.npcs[npc_id].name}.", "Reset the session to replay another route."]


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _quest_complete(required_steps: list[str], completed_steps: list[str]) -> bool:
    return all(step in completed_steps for step in required_steps)


def _extract_player_name(text: str) -> str | None:
    clean = text.strip()
    patterns = [
        r"(?:我是|我叫|叫我)\s*([A-Za-z一-鿿][A-Za-z一-鿿0-9_\- ]{0,24})",
        r"(?:my name is|i am|i'm|call me)\s+([A-Za-z][A-Za-z0-9_\- ]{0,24})",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if not match:
            continue
        name = match.group(1).strip(" ，。！？!?.,;:\"'")
        if name:
            return name[:32]
    return None
