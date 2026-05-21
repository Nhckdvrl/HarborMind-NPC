from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from game_npc_llm.data.prompts import npc_json_system_prompt, npc_json_user_prompt
from game_npc_llm.data.schemas import NPCAction, NPCResponse, repair_npc_response
from game_npc_llm.product.memory import InMemoryMemoryStore, MemoryStore
from game_npc_llm.product.models import ChatResult, GameState, WorldDefinition
from game_npc_llm.product.policy import PolicyClient, RulePolicyClient
from game_npc_llm.product.world import load_world

_REPEATED_PHRASES = ["不是问过", "不是問過", "问过了", "問過了"]
_LOCATION_PHRASES = ["这里是哪", "這裡是哪", "在哪", "哪儿", "哪裡"]


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
        return cls(
            world=load_world(),
            policy=RulePolicyClient(),
            memory=InMemoryMemoryStore(),
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

    def chat(self, npc_id: str, player_input: str, session_id: str = "demo") -> ChatResult:
        if npc_id not in self.world.npcs:
            raise ValueError(f"Unknown npc_id: {npc_id}")
        state = self.state_for(session_id)
        npc = self.world.npcs[npc_id]
        location = self.world.locations[npc.location_id]
        extracted_memories = self._extract_player_facts(state, npc_id, player_input)
        memory_hits = self.memory.search(session_id, npc_id, player_input)
        system = npc_json_system_prompt(
            persona=f"{npc.name}, {npc.role}. {npc.persona}",
            setting=self._setting_text(location),
            goal="; ".join(npc.goals),
        )
        user = npc_json_user_prompt(
            player_input,
            self._state_text(state, npc_id),
            memory_hits,
            self._response_guidance(state, npc_id, player_input),
        )
        if self.policy.__class__ is RulePolicyClient:
            raw = self.policy.complete_turn(self.world, state, npc_id, player_input, memory_hits)
        else:
            raw = self.policy.complete(
                [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
        response = repair_npc_response(raw)
        self._apply_contextual_dialogue_guard(state, npc_id, player_input, response)
        result = self._execute_response(state, npc_id, player_input, response)
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

    def _state_text(self, state: GameState, npc_id: str) -> str:
        location = self.world.locations[state.current_location_id]
        npc = self.world.npcs[npc_id]
        player_profile = state.player_profile or {}
        profile_text = ", ".join(f"{key}: {value}" for key, value in player_profile.items()) or "unknown"
        return "\n".join(
            [
                f"Current location: {location.name}",
                f"Speaking NPC: {npc.name}",
                f"Known player profile: {profile_text}",
                f"Inventory: {', '.join(state.inventory) or 'empty'}",
                f"Quest status: {state.quest_status}",
                f"Quest steps: {state.quest_steps}",
                f"Known clues: {state.known_clues or 'none'}",
                f"World flags: {state.world_flags or 'none'}",
                f"Relationship with {npc.name}: {state.relationships.get(npc_id, 0)}",
                f"Recent turns: {' | '.join(state.recent_turns[-6:]) or 'none'}",
                f"Allowed NPC actions: {[action.value for action in npc.allowed_actions]}",
            ]
        )

    def _setting_text(self, location) -> str:
        locations = "; ".join(
            f"{place.name} ({place.id}): {place.description}" for place in self.world.locations.values()
        )
        quests = "; ".join(f"{quest.title}: {quest.summary}" for quest in self.world.quests.values())
        entities = ", ".join(sorted(self.world.allowed_entities()))
        return "\n".join(
            [
                f"World: {self.world.title}. {self.world.premise}",
                f"Current NPC home location: {location.name}: {location.description}",
                f"Valid locations: {locations}",
                f"Active questlines: {quests}",
                f"Allowed world entities: {entities}",
            ]
        )

    def _extract_player_facts(self, state: GameState, npc_id: str, player_input: str) -> list[str]:
        del npc_id
        facts: list[str] = []
        name = _extract_player_name(player_input)
        if name:
            state.player_profile["name"] = name
            facts.append(f"The player's name is {name}. Do not ask for their name again.")
        return facts

    def _response_guidance(self, state: GameState, npc_id: str, player_input: str) -> str:
        npc = self.world.npcs[npc_id]
        location = self.world.locations[state.current_location_id]
        text = player_input.lower()
        hints = ["Answer the player's latest message directly; do not reset the conversation."]
        name = state.player_profile.get("name")
        if name:
            hints.append(f"The player already introduced themselves as {name}; do not ask their name again.")
        if any(phrase in player_input for phrase in ["你谁", "你是谁", "你是誰"]) or "who are you" in text:
            hints.append(f"Identify yourself as {npc.name}, {npc.role}, in one natural sentence.")
        if any(phrase in player_input for phrase in _REPEATED_PHRASES) or "already asked" in text:
            if name:
                hints.append(
                    f"Apologize briefly and acknowledge that you remember the player's name is {name}."
                )
            else:
                hints.append("Apologize briefly for repeating yourself and continue from the recent turns.")
        if any(phrase in player_input for phrase in _LOCATION_PHRASES) or "where am" in text or "where is this" in text:
            hints.append(
                f"Explain that this is {location.name} in {self.world.title}; do not mention any real-world city."
            )
        briefing = self._npc_briefing(npc, state, npc_id)
        if briefing:
            hints.append(briefing)
        return " ".join(hints)

    def _npc_briefing(self, npc, state: GameState, npc_id: str) -> str:
        relation = state.relationships.get(npc_id, 0)
        parts: list[str] = []
        if npc.refusal_policy:
            parts.append(f"Refusal policy: {npc.refusal_policy}")
        for secret_key, condition in npc.secret_conditions.items():
            threshold = npc.relationship_thresholds.get(secret_key)
            unlocked = threshold is None or relation >= threshold
            gate = (
                f" (needs relationship >= {threshold}, currently {relation})"
                if threshold is not None
                else ""
            )
            status = "you may reveal it now" if unlocked else "keep it hidden for now"
            parts.append(f"Secret '{secret_key}' unlocks when: {condition}{gate}; {status}.")
        if npc.secrets:
            parts.append(
                "Private facts you know but must guard until earned: " + " | ".join(npc.secrets)
            )
        return " ".join(parts)

    def _apply_contextual_dialogue_guard(
        self,
        state: GameState,
        npc_id: str,
        player_input: str,
        response: NPCResponse,
    ) -> None:
        npc = self.world.npcs[npc_id]
        npc_location = self.world.locations[npc.location_id]
        player_location = self.world.locations[state.current_location_id]
        text = player_input.lower()
        player_name = state.player_profile.get("name")
        chinese = _contains_cjk(player_input)

        if any(phrase in player_input for phrase in ["你谁", "你是谁", "你是誰"]) or "who are you" in text:
            response.dialogue = (
                _npc_identity_line(npc_id, npc.name, npc.role, npc_location.name)
                if chinese
                else f"I'm {npc.name}, {npc.role}, stationed at {npc_location.name}. {npc.persona}"
            )
            response.action = NPCAction.speak
            return

        if _extract_player_name(player_input):
            response.dialogue = (
                f"记住了，{player_name}。{_npc_identity_line(npc_id, npc.name, npc.role, npc_location.name)}"
                if chinese
                else f"Good to meet you, {player_name}. I'm {npc.name}, {npc.role}; ask me anything."
            )
            response.action = NPCAction.speak
            return

        if any(phrase in player_input for phrase in _REPEATED_PHRASES) or "already asked" in text:
            if player_name:
                response.dialogue = (
                    f"抱歉，{player_name}，我刚才重复了。我们接着刚才的话说。"
                    if chinese
                    else f"Sorry, {player_name}, I repeated myself. Let's continue from where we were."
                )
            else:
                response.dialogue = (
                    "抱歉，我刚才重复了。我们接着刚才的话说。"
                    if chinese
                    else "Sorry, I repeated myself. Let's continue from where we were."
                )
            response.action = NPCAction.speak
            return

        if any(phrase in player_input for phrase in _LOCATION_PHRASES) or "where am" in text or "where is this" in text:
            response.dialogue = (
                f"你现在在{self.world.title}的{player_location.name}。{player_location.description}"
                if chinese
                else f"This is {player_location.name} in {self.world.title}. {player_location.description}"
            )
            response.action = NPCAction.speak

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
        if any(term in text for term in ["please", "thanks", "thank", "help", "trust", "ありがとう"]):
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


def _quest_complete(required_steps: list[str], completed_steps: list[str]) -> bool:
    return all(step in completed_steps for step in required_steps)


def _extract_player_name(text: str) -> str | None:
    clean = text.strip()
    patterns = [
        r"(?:我是|我叫|叫我)\s*([A-Za-z\u4e00-\u9fff][A-Za-z\u4e00-\u9fff0-9_\- ]{0,24})",
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


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _npc_identity_line(npc_id: str, name: str, role: str, location_name: str) -> str:
    role_zh = {
        "mika": "港口机械师",
        "ren": "见习档案员",
        "hana": "神社守钟人",
        "toma": "航运公会协调人",
        "iko": "灯塔维护AI",
    }.get(npc_id, role)
    lines = {
        "mika": (
            f"我是{name}，{role_zh}，驻守在{location_name}；我只管潮汐引擎、压力阀、维修舱口和安全封条。"
        ),
        "ren": (
            f"我是{name}，{role_zh}，在{location_name}核对旧账本；我负责失踪账本、缩微胶片和被抹掉的靠港记录。"
        ),
        "hana": (
            f"我是{name}，{role_zh}，守着{location_name}的旧钟；我解读灯塔警告、船长之声和港口仪式。"
        ),
        "toma": (
            f"我是{name}，{role_zh}，常在{location_name}替公会收拾麻烦；除非你拿出证据，否则我不会承认靠港记录的事。"
        ),
        "iko": (
            f"我是{name}，{role_zh}，从{location_name}监测灯塔信号；我负责路线建议、传感器缺口和风险预警。"
        ),
    }
    return lines.get(npc_id, f"我是{name}，{role_zh}，驻守在{location_name}。")
