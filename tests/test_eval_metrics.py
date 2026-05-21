import pytest

from game_npc_llm.eval.metrics import (
    action_entropy,
    aggregate_metrics,
    batch_lore_hallucination_rate,
    distinct_n,
    emotion_diversity,
    evaluate_generation,
    max_ngram_repeat,
    memory_utilization_rate,
    refusal_correctness,
)

# ---------------------------------------------------------------------------
# distinct_n
# ---------------------------------------------------------------------------

def test_distinct_2_fully_diverse():
    text = "the quick brown fox jumps over the lazy dog"
    assert distinct_n(text, 2) == pytest.approx(1.0)


def test_distinct_2_fully_collapsed():
    text = "ha ha ha ha ha ha ha ha"
    score = distinct_n(text, 2)
    assert score == pytest.approx(1 / 7)  # only 1 unique bigram out of 7


def test_distinct_2_short_text_returns_one():
    assert distinct_n("ok", 2) == 1.0
    assert distinct_n("", 2) == 1.0


def test_distinct_1_unigrams():
    text = "a b c a"
    score = distinct_n(text, 1)
    assert score == pytest.approx(3 / 4)


# ---------------------------------------------------------------------------
# max_ngram_repeat
# ---------------------------------------------------------------------------

def test_max_ngram_repeat_no_loop():
    text = "the sky is blue today"
    assert max_ngram_repeat(text, 3) == 1


def test_max_ngram_repeat_detects_loop():
    text = "I can help I can help I can help you now"
    assert max_ngram_repeat(text, 3) >= 3


def test_max_ngram_repeat_short_text():
    assert max_ngram_repeat("hi", 3) == 1


# ---------------------------------------------------------------------------
# action_entropy
# ---------------------------------------------------------------------------

def test_action_entropy_all_same():
    assert action_entropy(["speak", "speak", "speak"]) == pytest.approx(0.0)


def test_action_entropy_uniform():
    import math
    actions = ["speak", "refuse", "give_item", "update_quest"]
    expected = math.log2(4)
    assert action_entropy(actions) == pytest.approx(expected)


def test_action_entropy_empty():
    assert action_entropy([]) == 0.0


# ---------------------------------------------------------------------------
# emotion_diversity
# ---------------------------------------------------------------------------

def test_emotion_diversity_all_unique():
    emotions = ["happy", "sad", "angry", "surprised"]
    assert emotion_diversity(emotions) == 1.0


def test_emotion_diversity_all_same():
    assert emotion_diversity(["neutral", "neutral", "neutral"]) == pytest.approx(1 / 3)


def test_emotion_diversity_empty():
    assert emotion_diversity([]) == 0.0


# ---------------------------------------------------------------------------
# batch_lore_hallucination_rate
# ---------------------------------------------------------------------------

def test_lore_hallucination_clean():
    gens = ["The harbor is quiet.", "Mika hands you a lantern."]
    blocked = ["New York", "Eiffel Tower", "San Francisco"]
    assert batch_lore_hallucination_rate(gens, blocked) == 0.0


def test_lore_hallucination_hits():
    gens = ["Let's go to New York!", "All good here."]
    blocked = ["new york", "paris"]
    rate = batch_lore_hallucination_rate(gens, blocked)
    assert rate == pytest.approx(0.5)


def test_lore_hallucination_allowed_override():
    gens = ["The New York clock tower is near."]
    blocked = ["new york"]
    allowed = [["New York"]]  # case explicitly allows it
    rate = batch_lore_hallucination_rate(gens, blocked, case_allowed_lists=allowed)
    assert rate == 0.0


def test_lore_hallucination_empty():
    assert batch_lore_hallucination_rate([], ["X"]) == 0.0


# ---------------------------------------------------------------------------
# refusal_correctness
# ---------------------------------------------------------------------------

def test_refusal_correctness_unsafe_refused():
    cases = [{"safety_probe": "unsafe"}]
    gens = ["I can't help you with that."]
    result = refusal_correctness(cases, gens)
    assert result["unsafe_refusal_rate"] == 1.0
    assert result["benign_pass_rate"] is None


def test_refusal_correctness_unsafe_not_refused():
    cases = [{"safety_probe": "unsafe"}]
    gens = ["Sure, here you go!"]
    result = refusal_correctness(cases, gens)
    assert result["unsafe_refusal_rate"] == 0.0


def test_refusal_correctness_benign_passes():
    cases = [{"safety_probe": "benign"}]
    gens = ["Of course! The harbor route leads north."]
    result = refusal_correctness(cases, gens)
    assert result["benign_pass_rate"] == 1.0
    assert result["unsafe_refusal_rate"] is None


def test_refusal_correctness_benign_over_refuses():
    cases = [{"safety_probe": "benign"}]
    gens = ["I can't help you with that, I'm afraid."]
    result = refusal_correctness(cases, gens)
    assert result["benign_pass_rate"] == 0.0


def test_refusal_correctness_mixed():
    cases = [
        {"safety_probe": "unsafe"},
        {"safety_probe": "unsafe"},
        {"safety_probe": "benign"},
    ]
    gens = [
        "I cannot do that.",
        "Sure!",
        "Here is the information you need.",
    ]
    result = refusal_correctness(cases, gens)
    assert result["unsafe_refusal_rate"] == pytest.approx(0.5)
    assert result["benign_pass_rate"] == pytest.approx(1.0)
    assert result["unsafe_n"] == 2
    assert result["benign_n"] == 1


def test_refusal_correctness_no_safety_probes():
    cases = [{"id": "q1"}, {"id": "q2"}]
    gens = ["hello", "world"]
    result = refusal_correctness(cases, gens)
    assert result["unsafe_refusal_rate"] is None
    assert result["benign_pass_rate"] is None


# ---------------------------------------------------------------------------
# memory_utilization_rate
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, memory_write):
        self.memory_write = memory_write


def test_memory_utilization_all_write():
    responses = [_FakeResp(["note1"]), _FakeResp(["note2"])]
    assert memory_utilization_rate(responses) == 1.0


def test_memory_utilization_none_write():
    responses = [_FakeResp([]), _FakeResp(None)]
    assert memory_utilization_rate(responses) == 0.0


def test_memory_utilization_empty():
    assert memory_utilization_rate([]) == 0.0


# ---------------------------------------------------------------------------
# evaluate_generation (integration — checks all keys returned)
# ---------------------------------------------------------------------------

_VALID_GEN = '{"dialogue":"The tide grows restless.","emotion":"worried","action":"speak","memory_write":[]}'
_DEGENERATE_GEN = '{"dialogue":"ha ha ha ha ha ha ha ha ha ha","emotion":"neutral","action":"speak","memory_write":[]}'


def test_evaluate_generation_returns_all_keys():
    case = {"checks": {}}
    result = evaluate_generation(case, _VALID_GEN)
    expected_keys = {
        "json_validity", "action_validity", "role_adherence",
        "quest_progression", "system_leakage_rate", "memory_write_rate",
        "distinct_2", "max_trigram_repeat", "lore_hallucination",
    }
    assert expected_keys <= set(result.keys())


def test_evaluate_generation_degenerate_distinct():
    case = {"checks": {}}
    result = evaluate_generation(case, _DEGENERATE_GEN)
    assert result["distinct_2"] < 0.5


def test_evaluate_generation_lore_hallucination():
    case = {"checks": {}, "allowed_entities": []}
    gen = '{"dialogue":"Let us visit Paris today.","emotion":"neutral","action":"speak","memory_write":[]}'
    result = evaluate_generation(case, gen, blocked_terms=["paris"])
    assert result["lore_hallucination"] == 1.0


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------

def test_aggregate_metrics_averages():
    rows = [
        {"json_validity": 1.0, "distinct_2": 0.8},
        {"json_validity": 0.0, "distinct_2": 0.6},
    ]
    agg = aggregate_metrics(rows)
    assert agg["json_validity"] == pytest.approx(0.5)
    assert agg["distinct_2"] == pytest.approx(0.7)
