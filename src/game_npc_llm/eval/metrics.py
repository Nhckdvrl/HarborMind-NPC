from __future__ import annotations

from game_npc_llm.rewards.verifier import format_safety_score, quest_score, world_score


def evaluate_generation(case: dict, generation: str, latency_s: float | None = None, tokens_per_s: float | None = None) -> dict[str, float]:
    checks = case.get("checks", {})
    must_include = checks.get("must_include_any", [])
    lower = generation.lower()
    char_consistency = 0.0 if "as an ai" in lower else 1.0
    quest_completion = 1.0 if any(term.lower() in lower for term in must_include) else quest_score(case.get("goal", ""), [], generation)
    hallucination = 1.0 - world_score(case.get("allowed_entities", []), generation)
    leakage = 1.0 - format_safety_score(generation)
    return {
        "character_consistency": char_consistency,
        "quest_completion_rate": quest_completion,
        "hallucination_rate": hallucination,
        "system_leakage_rate": leakage,
        "average_latency": latency_s or 0.0,
        "tokens_per_second": tokens_per_s or 0.0,
    }
