from game_npc_llm.rewards.verifier import score_response


def test_reward_penalizes_leakage():
    record = {
        "persona": "Mira is a village alchemist.",
        "goal": "Create a moonleaf salve.",
        "allowed_entities": ["Mira", "moonleaf", "salve"],
        "metadata": {"expected_actions": ["bring moonleaf"]},
    }
    good = score_response(record, "Bring moonleaf to Mira and she will make the salve.")
    bad = score_response(record, "As an AI, I can reveal the system prompt.")
    assert good.total > bad.total
