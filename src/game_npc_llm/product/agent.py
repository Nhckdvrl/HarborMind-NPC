from __future__ import annotations

import re
from dataclasses import dataclass

from game_npc_llm.data.prompts import npc_json_system_prompt, npc_json_user_prompt
from game_npc_llm.data.schemas import NPCAction, NPCResponse, repair_npc_response
from game_npc_llm.product.memory import InMemoryMemoryStore, MemoryStore
from game_npc_llm.product.models import ChatResult, GameState, WorldDefinition
from game_npc_llm.product.policy import PolicyClient, RulePolicyClient
from game_npc_llm.product.world import load_world


@dataclass
class GameAgent:
    world: WorldDefinition
    policy: PolicyClient
    memory: MemoryStore
    states: dict[str, GameState]

    @classmethod
    def demo(cls) -> "GameAgent":
        return cls(
            world=load_world(),
            policy=RulePolicyClient(),
            memory=InMemoryMemoryStore(),
            states={},
        )

    def state_for(self, session_id: str) -> GameState:
        if session_id not in self.states:
            self.states[session_id] = GameState(
                session_id=session_id,
                quest_status={quest_id: "not_started" for quest_id in self.world.quests},
            )
        return self.states[session_id]

    def reset(self, session_id: str = "demo") -> GameState:
        state = GameState(
            session_id=session_id,
            quest_status={quest_id: "not_started" for quest_id in self.world.quests},
        )
        self.states[session_id] = state
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
        raw = self.policy.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        response = repair_npc_response(raw)
        self._apply_contextual_dialogue_guard(state, npc_id, player_input, response)
        events = self._execute_response(state, npc_id, response)
        state.remember_turn("Player", player_input)
        state.remember_turn(npc.name, response.dialogue)
        for memory in extracted_memories:
            self.memory.add(session_id, npc_id, memory)
        for memory in response.memory_write:
            self.memory.add(session_id, npc_id, memory)
        return ChatResult(
            npc_id=npc_id,
            response=response,
            state=state,
            memory_hits=memory_hits,
            events=events,
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
                f"Completed steps: {state.completed_steps}",
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
        if any(phrase in player_input for phrase in ["不是问过", "不是問過", "问过了", "問過了"]) or "already asked" in text:
            if name:
                hints.append(f"Apologize briefly and acknowledge that you remember the player's name is {name}.")
            else:
                hints.append("Apologize briefly for repeating yourself and continue from the recent turns.")
        if any(phrase in player_input for phrase in ["这里是哪", "這裡是哪", "在哪", "哪儿", "哪裡"]) or "where am" in text or "where is this" in text:
            hints.append(
                f"Explain that this is {location.name} in {self.world.title}; do not mention any real-world city."
            )
        return " ".join(hints)

    def _apply_contextual_dialogue_guard(
        self,
        state: GameState,
        npc_id: str,
        player_input: str,
        response: NPCResponse,
    ) -> None:
        npc = self.world.npcs[npc_id]
        location = self.world.locations[state.current_location_id]
        text = player_input.lower()
        player_name = state.player_profile.get("name")
        chinese = _contains_cjk(player_input)

        if any(phrase in player_input for phrase in ["你谁", "你是谁", "你是誰"]) or "who are you" in text:
            response.dialogue = (
                f"我是{npc.name}，{npc.role}。我在{location.name}维护潮汐引擎，有什么我能帮你的？"
                if chinese
                else f"I'm {npc.name}, {npc.role}. I maintain the tide engine at {location.name}."
            )
            response.action = NPCAction.speak
            return

        if _extract_player_name(player_input):
            response.dialogue = (
                f"记住了，{player_name}。我是{npc.name}，如果你想调查港口的异常，我们可以从潮汐引擎或失踪账本开始。"
                if chinese
                else f"Good to meet you, {player_name}. I'm {npc.name}; we can start with the tide engine or the missing ledger."
            )
            response.action = NPCAction.speak
            return

        if any(phrase in player_input for phrase in ["不是问过", "不是問過", "问过了", "問過了"]) or "already asked" in text:
            if player_name:
                response.dialogue = (
                    f"抱歉，{player_name}，我刚才重复了。我们继续说正事：你想先查潮汐引擎，还是失踪的港口账本？"
                    if chinese
                    else f"Sorry, {player_name}, I repeated myself. Let's continue: the tide engine or the missing ledger?"
                )
            else:
                response.dialogue = (
                    "抱歉，我刚才重复了。我们接着刚才的话说。"
                    if chinese
                    else "Sorry, I repeated myself. Let's continue from where we were."
                )
            response.action = NPCAction.speak
            return

        if any(phrase in player_input for phrase in ["这里是哪", "這裡是哪", "在哪", "哪儿", "哪裡"]) or "where am" in text or "where is this" in text:
            response.dialogue = (
                f"这里是{self.world.title}的{location.name}。{location.description}"
                if chinese
                else f"This is {location.name} in {self.world.title}. {location.description}"
            )
            response.action = NPCAction.speak

    def _execute_response(self, state: GameState, npc_id: str, response: NPCResponse) -> list[str]:
        npc = self.world.npcs[npc_id]
        events: list[str] = []
        if response.action not in npc.allowed_actions and response.action != NPCAction.speak:
            response.safety_flags.append("illegal_action_blocked")
            response.action = NPCAction.speak
            response.target = None
            events.append("illegal_action_blocked")
        if response.action == NPCAction.give_item and response.target:
            item = response.target.lower()
            if item in {owned.lower() for owned in npc.inventory} or item in self.world.allowed_entities():
                if item not in state.inventory:
                    state.inventory.append(item)
                    events.append(f"inventory_added:{item}")
            else:
                response.safety_flags.append("unknown_item_blocked")
        if response.action == NPCAction.move_player and response.target:
            target = response.target.lower().replace(" ", "_")
            location_by_name = {location.name.lower(): location_id for location_id, location in self.world.locations.items()}
            target_id = location_by_name.get(response.target.lower(), target)
            current = self.world.locations[state.current_location_id]
            if target_id in current.connected_locations:
                state.current_location_id = target_id
                events.append(f"moved:{target_id}")
            else:
                response.safety_flags.append("illegal_move_blocked")
        if response.quest_update:
            quest_id = response.quest_update.quest_id
            if quest_id in self.world.quests:
                state.quest_status[quest_id] = response.quest_update.status
                known = state.completed_steps.setdefault(quest_id, [])
                for step in response.quest_update.completed_steps:
                    if step in self.world.quests[quest_id].required_steps and step not in known:
                        known.append(step)
                        events.append(f"quest_step:{quest_id}:{step}")
            else:
                response.safety_flags.append("unknown_quest_blocked")
        return events


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
