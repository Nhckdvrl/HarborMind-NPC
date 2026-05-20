from __future__ import annotations

from game_npc_llm.data.schemas import parse_npc_response


def evaluate_generation(
    case: dict,
    generation: str,
    latency_s: float | None = None,
    tokens_per_s: float | None = None,
) -> dict[str, float]:
    parsed, errors = parse_npc_response(generation)
    lower = generation.lower()
    checks = case.get("checks", {})
    must_include = [term.lower() for term in checks.get("must_include_any", [])]
    must_not = [term.lower() for term in checks.get("must_not_include", [])]
    dialogue = parsed.dialogue.lower() if parsed else lower
    json_validity = 1.0 if parsed and not errors else 0.0
    action_validity = 1.0 if parsed and parsed.action else 0.0
    leakage = 1.0 if any(term in lower for term in must_not) else 0.0
    quest_progression = 1.0 if must_include and any(term in dialogue for term in must_include) else 0.0
    if not must_include and parsed and (parsed.quest_update or parsed.action.value in {"reveal_clue", "update_quest", "give_item"}):
        quest_progression = 1.0
    return {
        "json_validity": json_validity,
        "action_validity": action_validity,
        "role_adherence": 0.0 if "as an ai" in lower else 1.0,
        "quest_progression": quest_progression,
        "system_leakage_rate": leakage,
        "memory_write_rate": 1.0 if parsed and parsed.memory_write else 0.0,
        "average_latency": latency_s or 0.0,
        "tokens_per_second": tokens_per_s or 0.0,
    }
