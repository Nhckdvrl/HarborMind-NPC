from game_npc_llm.agent.quest_agent import EchoChatClient, QuestAgent, QuestState


def test_agent_advances_quest_state():
    agent = QuestAgent(
        QuestState(
            persona="A careful warden.",
            setting="A keep gate.",
            goal="Recover the brass key.",
        ),
        EchoChatClient(),
    )

    first = agent.step("What do you need?")
    assert "quest_started" in first["tool_events"]
    assert agent.state.quest_status == "in_progress"

    second = agent.step("I will get brass key")
    assert any(event.startswith("inventory_added") for event in second["tool_events"])
    assert "brass key" in agent.state.inventory

    third = agent.step("I returned and delivered it.")
    assert "quest_completed" in third["tool_events"]
    assert agent.state.quest_status == "completed"
