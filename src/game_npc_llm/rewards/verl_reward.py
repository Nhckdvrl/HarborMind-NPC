from __future__ import annotations

from typing import Any

from game_npc_llm.rewards.verifier import score_response


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any | None = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    """verl custom reward entrypoint for rule-based NPC quest scoring."""
    record = dict(extra_info or {})
    if ground_truth and not record.get("goal"):
        record["goal"] = str(ground_truth)
    if data_source and not record.get("source"):
        record["source"] = data_source
    return float(score_response(record, solution_str or "").total)


def reward_from_sample(sample: dict[str, Any]) -> float:
    """Small test/helper wrapper for records shaped like verl parquet rows."""
    reward_model = sample.get("reward_model") or {}
    return compute_score(
        data_source=sample.get("data_source"),
        solution_str=sample.get("response") or sample.get("solution_str") or "",
        ground_truth=reward_model.get("ground_truth"),
        extra_info=sample.get("extra_info") or {},
    )
