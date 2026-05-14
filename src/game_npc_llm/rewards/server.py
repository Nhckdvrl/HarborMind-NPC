from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from game_npc_llm.rewards.judge import judge_response
from game_npc_llm.rewards.verifier import score_response

app = FastAPI(title="Game NPC Reward Server")


class RewardRequest(BaseModel):
    prompt: dict[str, Any]
    responses: list[str]
    use_llm_judge: bool = False


@app.post("/reward")
def reward(request: RewardRequest) -> dict[str, Any]:
    rewards = []
    for response in request.responses:
        judge_score = None
        judge_reason = None
        if request.use_llm_judge:
            judge_score, judge_reason = judge_response(request.prompt, response)
        breakdown = score_response(request.prompt, response, judge_score)
        rewards.append({"reward": breakdown.total, "breakdown": breakdown.to_dict(), "judge_reason": judge_reason})
    return {"rewards": rewards}
