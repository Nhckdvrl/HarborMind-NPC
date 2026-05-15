from __future__ import annotations

from typing import Any

from game_npc_llm.rewards.judge import judge_response
from game_npc_llm.rewards.verifier import score_response


async def reward_func(args: Any, sample: Any, **_: Any) -> float:
    metadata = getattr(sample, "metadata", None) or {}
    response = str(getattr(sample, "response", "") or "")
    judge_score = None

    if bool(getattr(args, "npc_use_llm_judge", False)):
        judge_score, _ = judge_response(metadata, response)

    return float(score_response(metadata, response, judge_score).total)
