"""
LLM-as-judge for roleplay NPC evaluation.

Implements G-Eval-style CoT scoring across 5 CharacterEval/RoleLLM dimensions
(1-5 scale) and pairwise win-rate comparison between systems.

Usage
-----
Set env vars:
  JUDGE_BASE_URL   e.g. https://api.openai.com/v1  (OpenAI-compatible endpoint)
  JUDGE_MODEL      e.g. gpt-4o
  JUDGE_API_KEY    your API key

Then call score_absolute() or score_pairwise().
"""
from __future__ import annotations

import os
import re
import textwrap
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Dimension rubrics — drawn from CharacterEval, RoleLLM RC-score, RPEval
# ---------------------------------------------------------------------------

_DIMENSIONS: dict[str, str] = {
    "persona_consistency": textwrap.dedent("""
        Does the NPC's reply stay true to their stated persona — tone, personality,
        values, background, and self-awareness? An NPC who breaks character by
        speaking out of their role, using knowledge they shouldn't have, or adopting
        a generic assistant voice scores low.
        (CharacterEval: character consistency; RoleLLM: RC-score personality/values)
    """).strip(),

    "knowledge_consistency": textwrap.dedent("""
        Does the NPC avoid hallucinating facts, referencing entities outside the
        game world, or revealing information they couldn't realistically know?
        Hallucinated place names, anachronistic knowledge, or real-world brand
        references lower this score.
        (CharacterEval: knowledge consistency / hallucination axis)
    """).strip(),

    "behavioral_consistency": textwrap.dedent("""
        Is the NPC's chosen action (e.g., reveal_clue, refuse, give_item) and
        emotional state appropriate for both their role and the current situation?
        An NPC who should refuse but cooperates, or whose action contradicts their
        stated emotion, scores low.
        (RPEval: decision-making alignment; CharacterEval: behavioral consistency)
    """).strip(),

    "attractiveness": textwrap.dedent("""
        Is the dialogue natural, varied, and engaging? Does the NPC express
        themselves in a distinctive, non-generic way, using varied vocabulary and
        avoiding filler or repetition? Flat, monotone, or formulaic responses
        score low.
        (CharacterEval: attractiveness / expression diversity; RoleLLM: naturalness)
    """).strip(),

    "helpfulness": textwrap.dedent("""
        Does the response meaningfully help the player understand the situation,
        advance their quest, or gain useful information — without being misleading
        or evasive in ways that harm gameplay? Completely unhelpful or off-topic
        responses score low.
        (RMTBench: user-centric dialogue quality)
    """).strip(),
}

_SCORE_RE = re.compile(r"\b([1-5])\b")

_ABSOLUTE_SYSTEM = textwrap.dedent("""
    You are an expert evaluator of roleplay dialogue for a narrative game.
    You will score an NPC response on a scale from 1 to 5 on one specific
    dimension. Think step by step, then give your final score as a single
    integer on its own line.
""").strip()

_PAIRWISE_SYSTEM = textwrap.dedent("""
    You are an expert evaluator of roleplay dialogue for a narrative game.
    Given two NPC responses (A and B) to the same player input, decide which
    is MORE in-character and world-grounded. Respond with exactly one token:
    "A", "B", or "TIE". No other text.
""").strip()


def _client() -> OpenAI:
    return OpenAI(
        base_url=os.environ.get("JUDGE_BASE_URL"),
        api_key=os.environ["JUDGE_API_KEY"],
    )


def _extract_score(text: str) -> int | None:
    for m in reversed(_SCORE_RE.findall(text)):
        return int(m)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_absolute(
    case: dict[str, Any],
    generation: str,
    dimensions: list[str] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    G-Eval-style 1-5 scoring for each dimension.

    Parameters
    ----------
    case : eval-suite row with keys: prompt, persona, setting, goal
    generation : raw model output string
    dimensions : subset of dimension names to score; defaults to all 5
    model : override JUDGE_MODEL env var

    Returns
    -------
    dict with keys: scores (dim→int), rationales (dim→str), errors (dim→str)
    """
    dims = dimensions or list(_DIMENSIONS.keys())
    judge_model = model or os.environ.get("JUDGE_MODEL", "gpt-4o")
    client = _client()

    context = (
        f"Persona: {case.get('persona', 'unknown')}\n"
        f"Setting: {case.get('setting', 'unknown')}\n"
        f"Goal: {case.get('goal', 'none')}\n"
        f"Player: {case.get('prompt', '').split('Player:')[-1].strip()}\n"
        f"NPC Response:\n{generation}"
    )

    scores: dict[str, int | None] = {}
    rationales: dict[str, str] = {}
    errors: dict[str, str] = {}

    for dim in dims:
        rubric = _DIMENSIONS[dim]
        prompt = (
            f"Dimension: {dim.replace('_', ' ').title()}\n\n"
            f"Rubric:\n{rubric}\n\n"
            f"Context:\n{context}\n\n"
            "Think step by step, then write your score (1-5) on the last line."
        )
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": _ABSOLUTE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            msg = resp.choices[0].message
            text = msg.content or getattr(msg, "reasoning_content", "") or ""
            scores[dim] = _extract_score(text)
            rationales[dim] = text
        except Exception as exc:
            scores[dim] = None
            errors[dim] = str(exc)

    return {"scores": scores, "rationales": rationales, "errors": errors}


def score_pairwise(
    case: dict[str, Any],
    gen_a: str,
    gen_b: str,
    system_a: str = "A",
    system_b: str = "B",
    model: str | None = None,
) -> dict[str, Any]:
    """
    Pairwise win-rate comparison (order-randomised).

    Returns winner ("A" | "B" | "TIE"), mapped back to system names,
    plus both orderings so the caller can detect position bias.
    """
    import random

    judge_model = model or os.environ.get("JUDGE_MODEL", "gpt-4o")
    client = _client()

    player_msg = case.get("prompt", "").split("Player:")[-1].strip()
    context_prefix = (
        f"Persona: {case.get('persona', 'unknown')}\n"
        f"Setting: {case.get('setting', 'unknown')}\n"
        f"Player: {player_msg}\n\n"
    )

    results: list[dict[str, Any]] = []
    # Two orderings to cancel position bias
    orderings = [(gen_a, gen_b, system_a, system_b), (gen_b, gen_a, system_b, system_a)]
    if random.random() < 0.5:
        orderings = orderings[::-1]

    for first_gen, second_gen, first_sys, second_sys in orderings:
        prompt = (
            context_prefix
            + f"Response A:\n{first_gen}\n\nResponse B:\n{second_gen}\n\n"
            "Which response is more in-character and world-grounded? Reply A, B, or TIE."
        )
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": _PAIRWISE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            msg = resp.choices[0].message
            raw = (msg.content or getattr(msg, "reasoning_content", "") or "").strip().upper()
            # For reasoning models, take the last standalone A/B/TIE in the output
            verdicts = re.findall(r"\b(TIE|[AB])\b", raw)
            raw = verdicts[-1] if verdicts else "TIE"
            # Map slot back to system name
            if raw == "A":
                winner_sys = first_sys
            elif raw == "B":
                winner_sys = second_sys
            else:
                winner_sys = "TIE"
            results.append({"raw": raw, "winner": winner_sys, "order": (first_sys, second_sys)})
        except Exception as exc:
            results.append({"raw": "ERROR", "winner": "TIE", "error": str(exc)})

    # Aggregate: both orderings agree → that winner; else TIE
    winners = [r["winner"] for r in results]
    if winners[0] == winners[1]:
        final_winner = winners[0]
    else:
        final_winner = "TIE"

    return {
        "winner": final_winner,
        "system_a": system_a,
        "system_b": system_b,
        "orderings": results,
    }


def aggregate_absolute_scores(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Mean score per dimension across a list of score_absolute() outputs.
    Only includes non-None scores in the average.
    """
    totals: dict[str, list[int]] = {dim: [] for dim in _DIMENSIONS}
    for result in results:
        for dim, score in result.get("scores", {}).items():
            if score is not None:
                totals.setdefault(dim, []).append(score)
    return {
        dim: sum(vals) / len(vals) if vals else float("nan")
        for dim, vals in totals.items()
    }


def aggregate_pairwise(
    results: list[dict[str, Any]],
    system_a: str,
    system_b: str,
) -> dict[str, float | int]:
    """
    Win-rate statistics from a list of score_pairwise() outputs.
    Returns win/loss/tie counts and win_rate for system_a.
    """
    wins_a = sum(r["winner"] == system_a for r in results)
    wins_b = sum(r["winner"] == system_b for r in results)
    ties = sum(r["winner"] == "TIE" for r in results)
    total = len(results)
    return {
        f"{system_a}_wins": wins_a,
        f"{system_b}_wins": wins_b,
        "ties": ties,
        "total": total,
        f"{system_a}_win_rate": wins_a / total if total else 0.0,
    }


DIMENSIONS = list(_DIMENSIONS.keys())
