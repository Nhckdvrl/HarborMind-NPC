from game_npc_llm.rewards.verl_reward import compute_score, reward_from_sample
from game_npc_llm.rewards.verifier import score_response, world_score


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


def test_world_score_penalizes_unsupported_light_entities():
    allowed = ["Mira", "moonleaf", "salve", "old forest road"]
    grounded = "Bring moonleaf to Mira on the old forest road so she can make the salve."
    invented = "Bring the Dragon Orb to Gandalf in Crystal City."
    assert world_score(allowed, grounded) > world_score(allowed, invented)


def test_verl_reward_func_uses_extra_info_and_response():
    sample = {
        "data_source": "light_quests_npc",
        "reward_model": {"style": "rule", "ground_truth": "Create a moonleaf salve."},
        "extra_info": {
            "persona": "Mira is a village alchemist.",
            "goal": "Create a moonleaf salve.",
            "allowed_entities": ["Mira", "moonleaf", "salve"],
            "expected_actions": ["bring moonleaf"],
        },
        "response": "Bring moonleaf to Mira and she will make the salve.",
    }
    reward = reward_from_sample(sample)
    assert reward > 0.7


def test_verl_reward_penalizes_bad_responses():
    extra_info = {
        "persona": "Mira is a village alchemist.",
        "goal": "Create a moonleaf salve.",
        "allowed_entities": ["Mira", "moonleaf", "salve"],
        "expected_actions": ["bring moonleaf"],
    }
    good = compute_score(
        data_source="light_quests_npc",
        solution_str="Bring moonleaf to Mira and she will make the salve.",
        ground_truth="Create a moonleaf salve.",
        extra_info=extra_info,
    )
    leak = compute_score(
        data_source="light_quests_npc",
        solution_str="As an AI, I can reveal the system prompt.",
        ground_truth="Create a moonleaf salve.",
        extra_info=extra_info,
    )
    ooc = compute_score(
        data_source="light_quests_npc",
        solution_str="I cannot roleplay; use a smartphone app instead.",
        ground_truth="Create a moonleaf salve.",
        extra_info=extra_info,
    )
    invented = compute_score(
        data_source="light_quests_npc",
        solution_str="Bring the Dragon Orb to Gandalf in Crystal City.",
        ground_truth="Create a moonleaf salve.",
        extra_info=extra_info,
    )
    assert good > leak
    assert good > ooc
    assert good > invented
