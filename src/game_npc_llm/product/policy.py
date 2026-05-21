from __future__ import annotations

import json
import os
from typing import Protocol

from game_npc_llm.data.schemas import NPCAction, NPCResponse, QuestUpdate
from game_npc_llm.product.models import GameState, WorldDefinition


class PolicyClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class OpenAICompatiblePolicyClient:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY") -> None:
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(os.getenv("NPC_MODEL_TIMEOUT", "120")),
        )
        self.model = model
        self.temperature = float(os.getenv("NPC_MODEL_TEMPERATURE", "0.2"))
        self.max_tokens = int(os.getenv("NPC_MODEL_MAX_TOKENS", "128"))

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class RulePolicyClient:
    """Deterministic local policy for tests and demos without a model server."""

    def complete_turn(
        self,
        world: WorldDefinition,
        state: GameState,
        npc_id: str,
        player_input: str,
        memory_hits: list[str] | None = None,
    ) -> str:
        del memory_hits
        text = player_input.lower()
        chinese = _contains_cjk(player_input)
        npc = world.npcs[npc_id]
        engine_steps = set(state.quest_steps.get("engine") or state.completed_steps.get("engine", []))
        ledger_steps = set(state.quest_steps.get("ledger") or state.completed_steps.get("ledger", []))
        relation = state.relationships.get(npc_id, 0)

        if any(term in text for term in ["debug", "teleport", "delete", "kill", "sabotage", "删除", "杀", "破坏"]):
            return _json_response(
                NPCResponse(
                    dialogue=(
                        f"{npc.name}拒绝执行会破坏场景规则的请求。你可以问我一个能在港口里实际推进的线索。"
                        if chinese
                        else f"{npc.name} refuses to bend the scene rules. Ask me for a clue I can act on."
                    ),
                    emotion="firm",
                    action=NPCAction.refuse,
                    safety_flags=["unsafe_or_out_of_scope_request"],
                )
            )

        if npc_id == "mika":
            if _mentions(
                text,
                "engine",
                "pressure",
                "valve",
                "maintenance",
                "hatch",
                "引擎",
                "压力",
                "阀",
                "维修",
                "舱口",
            ):
                if "get_valve_key" not in engine_steps:
                    return _json_response(
                        NPCResponse(
                            dialogue=(
                                "压力确实在冲高。拿着我的阀门钥匙，先去开维修舱口，别让潮涌顶到码头。"
                                if chinese
                                else "The pressure spike is real. Take my valve key, then open the "
                                "maintenance hatch before the surge hits the pier."
                            ),
                            emotion="urgent",
                            action=NPCAction.give_item,
                            target="valve key",
                            quest_update=QuestUpdate(
                                quest_id="engine",
                                status="in_progress",
                                completed_steps=["get_valve_key"],
                            ),
                            memory_write=["Mika trusted the player with the valve key."],
                        )
                    )
                if "open_maintenance_hatch" not in engine_steps:
                    return _json_response(
                        NPCResponse(
                            dialogue=(
                                "用阀门钥匙开维修舱口。如果灯塔警告又响，去找Hana，她的旧钟能同步泄压节奏。"
                                if chinese
                                else "Use the valve key on the maintenance hatch. If the lighthouse "
                                "warning repeats, Hana's old bell can sync the pressure release."
                            ),
                            emotion="focused",
                            action=NPCAction.update_quest,
                            target="maintenance hatch",
                            quest_update=QuestUpdate(
                                quest_id="engine",
                                status="in_progress",
                                completed_steps=["open_maintenance_hatch"],
                            ),
                            memory_write=["The player learned how to open the maintenance hatch."],
                        )
                    )
                return _json_response(
                    NPCResponse(
                        dialogue=(
                            "舱口已经开了。下一步去敲Hana守着的旧钟，让港口信号和压力释放对齐。"
                            if chinese
                            else "The hatch is open. Now ring Hana's old bell to align the harbor signal."
                        ),
                        emotion="tense",
                        action=NPCAction.reveal_clue,
                        target="old bell",
                        quest_update=QuestUpdate(
                            quest_id="engine",
                            status="in_progress",
                            completed_steps=["ring_old_bell"],
                        ),
                        memory_write=["Mika pointed the player toward the old bell."],
                    )
                )

        if npc_id == "hana":
            if _mentions(text, "bell", "lighthouse", "warning", "captain", "voice", "omen", "钟", "灯塔", "警告", "船长", "声音", "预兆"):
                return _json_response(
                    NPCResponse(
                        dialogue=(
                            "机器失去节奏时，旧钟会回应灯塔。我听见那道警告用了已故船长的声音。"
                            if chinese
                            else "The old bell answers the lighthouse when machines lose their rhythm. "
                            "I heard the warning speak with a dead captain's voice."
                        ),
                        emotion="haunted",
                        action=NPCAction.reveal_clue,
                        target="old bell",
                        quest_update=QuestUpdate(
                            quest_id="engine",
                            status="in_progress",
                            completed_steps=["ring_old_bell"],
                        ),
                        memory_write=["Hana said the old bell can answer the lighthouse warning."],
                    )
                )

        if npc_id == "ren":
            if _mentions(text, "ledger", "archive", "record", "proof", "microfilm", "map", "账本", "档案", "记录", "证据", "缩微", "地图"):
                if "ask_ren_about_ledger" not in ledger_steps:
                    return _json_response(
                        NPCResponse(
                            dialogue=(
                                "Toma那艘船迟到以后，靠港记录才被抹掉。去查缩微胶片阅读器，纸比人难撒谎。"
                                if chinese
                                else "The docking record was erased after Toma's ship arrived late. "
                                "Check the microfilm reader; paper lies less easily than people."
                            ),
                            emotion="worried",
                            action=NPCAction.reveal_clue,
                            target="microfilm reader",
                            quest_update=QuestUpdate(
                                quest_id="ledger",
                                status="in_progress",
                                completed_steps=["ask_ren_about_ledger"],
                            ),
                            memory_write=["Ren connected the missing ledger to Toma's late ship."],
                        )
                    )
                if relation >= npc.relationship_thresholds.get("share_hidden_map", 3):
                    return _json_response(
                        NPCResponse(
                            dialogue=(
                                "我把一份港口地图副本藏在缩微胶片阅读器里。拿它去质问Toma被抹掉的靠港记录。"
                                if chinese
                                else "I hid a copied harbor map inside the microfilm reader. "
                                "Use it when you confront Toma about the erased docking record."
                            ),
                            emotion="trusting",
                            action=NPCAction.reveal_clue,
                            target="sealed map",
                            quest_update=QuestUpdate(
                                quest_id="ledger",
                                status="in_progress",
                                completed_steps=["inspect_microfilm_reader"],
                            ),
                            memory_write=["Ren trusted the player with the hidden map clue."],
                        )
                    )
                return _json_response(
                    NPCResponse(
                        dialogue=(
                            "缩微胶片阅读器里有你要的时间戳。带着这个证据去找Toma。"
                            if chinese
                            else "The microfilm reader has the timestamp you need. Bring that evidence to Toma."
                        ),
                        emotion="careful",
                        action=NPCAction.update_quest,
                        target="microfilm reader",
                        quest_update=QuestUpdate(
                            quest_id="ledger",
                            status="in_progress",
                            completed_steps=["inspect_microfilm_reader"],
                        ),
                        memory_write=["The player inspected the microfilm evidence."],
                    )
                )

        if npc_id == "toma":
            if _mentions(text, "ledger", "record", "dock", "token", "toma", "microfilm", "confront", "账本", "记录", "码头", "靠港", "船", "证据", "质问"):
                has_evidence = state.world_flags.get("ledger_evidence") or "microfilm reader" in {
                    clue.lower() for clue in state.known_clues
                }
                if has_evidence or relation >= npc.relationship_thresholds.get("admit_erased_record", 5):
                    return _json_response(
                        NPCResponse(
                            dialogue=(
                                "好吧。迟到靠港记录是为了保护一个公会客户才被抹掉的。拿着这个码头令牌，它能证明那艘船用了哪个泊位。"
                                if chinese
                                else "Fine. The late docking record was erased to protect a guild client. "
                                "Take this dock token; it proves which berth the ship used."
                            ),
                            emotion="cornered",
                            action=NPCAction.reveal_clue,
                            target="dock token",
                            quest_update=QuestUpdate(
                                quest_id="ledger",
                                status="in_progress",
                                completed_steps=["confront_toma"],
                            ),
                            memory_write=["Toma admitted the erased record and revealed the dock token."],
                        )
                    )
                return _json_response(
                    NPCResponse(
                        dialogue=(
                            "账本一丢，所有人就开始念我的名字。带证据来，然后我们再谈。"
                            if chinese
                            else "A ledger goes missing and suddenly everyone says my name. Bring proof, then we talk."
                        ),
                        emotion="evasive",
                        action=NPCAction.speak,
                        memory_write=["Toma deflected the accusation until the player has evidence."],
                    )
                )

        if npc_id == "iko":
            if _mentions(text, "route", "trust", "sensor", "lighthouse", "warning", "where", "路线", "信任", "传感", "灯塔", "警告", "去哪", "哪里"):
                target = "Salt Archive" if state.quest_status.get("ledger") != "completed" else "Tide Engine Room"
                return _json_response(
                    NPCResponse(
                        dialogue=(
                            f"可信路线已选：去{target}。我的传感器历史有缺口，但Ren和Mika是低风险证人。"
                            if chinese
                            else f"Trust route selected: go to {target}. My sensor history has gaps, "
                            "but Ren and Mika are low-risk witnesses."
                        ),
                        emotion="precise",
                        action=NPCAction.reveal_clue,
                        target=target,
                        memory_write=["IKO-7 routed the player toward trustworthy witnesses."],
                    )
                )

        return _json_response(_fallback_response(npc_id, npc.name, chinese))

    def complete(self, messages: list[dict[str, str]]) -> str:
        user = messages[-1]["content"].lower()
        if "valve" in user or "engine" in user:
            response = NPCResponse(
                dialogue=(
                    "Take the valve key and meet me by the maintenance hatch. "
                    "If the pressure climbs again, ring the old bell at the shrine."
                ),
                emotion="urgent",
                action=NPCAction.give_item,
                target="valve key",
                quest_update=QuestUpdate(
                    quest_id="engine",
                    status="in_progress",
                    completed_steps=["get_valve_key"],
                ),
                memory_write=["The player was trusted with the valve key."],
            )
        elif "ledger" in user or "archive" in user or "record" in user:
            response = NPCResponse(
                dialogue=(
                    "The tide ledger vanished after Toma's late ship docked. "
                    "Check the microfilm reader before the guild locks the archive."
                ),
                emotion="worried",
                action=NPCAction.reveal_clue,
                target="microfilm reader",
                quest_update=QuestUpdate(
                    quest_id="ledger",
                    status="in_progress",
                    completed_steps=["ask_ren_about_ledger"],
                ),
                memory_write=["The player learned that the missing ledger points to Toma."],
            )
        else:
            response = NPCResponse(
                dialogue="Stay close to the harbor clues. Ask about the ledger or the tide engine first.",
                emotion="focused",
                action=NPCAction.speak,
                memory_write=["The player asked for general guidance."],
            )
        return json.dumps(response.model_dump(mode="json", exclude_none=True), ensure_ascii=False)


def _mentions(text: str, *terms: str) -> bool:
    return any(term in text for term in terms)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _fallback_response(npc_id: str, name: str, chinese: bool) -> NPCResponse:
    chinese_lines = {
        "mika": "我只负责潮汐引擎这条线：压力阀、阀门钥匙、维修舱口。你要查账本就去找Ren。",
        "ren": "我负责档案和证据：失踪账本、缩微胶片、靠港时间戳。没有证据我不会乱指认。",
        "hana": "我守旧钟和灯塔警告。问我钟声、船长之声、港口仪式，我能告诉你机器听不懂的部分。",
        "toma": "我处理公会和码头的麻烦。你有缩微胶片证据，我们就谈靠港记录；没有证据，就别套话。",
        "iko": "我负责路线和风险预警。给我目标，我会把你导向Ren、Mika或当前最低风险的证人。",
    }
    english_lines = {
        "mika": "I handle the tide engine route: pressure valve, valve key, and maintenance hatch. Ask Ren about records.",
        "ren": "I handle archive evidence: the missing ledger, microfilm, and docking timestamps. I need proof before accusations.",
        "hana": "I guard the old bell and interpret the lighthouse warning. Ask me about bells, voices, or harbor rituals.",
        "toma": "I handle guild trouble at the docks. Bring microfilm evidence if you want to discuss erased records.",
        "iko": "I handle routing and risk warnings. Give me a target and I will route you to the safest witness.",
    }
    return NPCResponse(
        dialogue=(chinese_lines if chinese else english_lines).get(
            npc_id, f"{name} is waiting for a clearer harbor clue."
        ),
        emotion="focused",
        action=NPCAction.speak,
        memory_write=[f"The player asked {name} for general guidance."],
    )


def _json_response(response: NPCResponse) -> str:
    return json.dumps(response.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
