from game_npc_llm.data.schemas import NPCAction
from game_npc_llm.product.agent import GameAgent
from game_npc_llm.product.memory import HashEmbeddingMemoryStore, InMemoryMemoryStore
from game_npc_llm.product.policy import RulePolicyClient
from game_npc_llm.product.world import load_world


class CapturePolicy:
    """Non-rule policy that records every complete() call for assertion."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return '{"dialogue": "测试回复", "emotion": "neutral", "action": "speak", "memory_write": []}'


class IllegalActionPolicy:
    def complete(self, messages):
        return (
            '{"dialogue":"I warp you into the debug room.","emotion":"excited",'
            '"action":"move_player","target":"developer console"}'
        )


def test_demo_agent_updates_quest_and_memory():
    agent = GameAgent.demo()
    result = agent.chat("mika", "The tide engine pressure is climbing.", session_id="test")

    assert result.response.action == NPCAction.give_item
    assert "valve key" in result.state.inventory
    assert result.state.quest_status["engine"] == "in_progress"
    assert result.inventory_delta == ["valve key"]
    assert result.relationship_delta["mika"] > 0
    assert "get_valve_key" in result.state.quest_steps["engine"]
    assert agent.memory.search("test", "mika", "valve key")


def test_agent_blocks_unknown_npc():
    agent = GameAgent.demo()
    try:
        agent.chat("missing", "hello")
    except ValueError as exc:
        assert "Unknown npc_id" in str(exc)
    else:
        raise AssertionError("missing NPC should fail")


def test_agent_remembers_player_name_in_prompt_and_memory():
    """Player name extracted from intro turn must appear in subsequent LLM context."""
    policy = CapturePolicy()
    agent = GameAgent(
        world=load_world(),
        policy=policy,
        memory=InMemoryMemoryStore(),
        states={},
    )

    agent.chat("mika", "我是湘婷", session_id="name-test")
    agent.chat("mika", "你好啊", session_id="name-test")

    state = agent.state_for("name-test")
    assert state.player_profile["name"] == "湘婷"

    # Agent goes through LLM path (CapturePolicy is not RulePolicyClient)
    assert len(policy.calls) == 2
    last_system = policy.calls[-1][0]["content"]
    last_user = policy.calls[-1][-1]["content"]

    # New agent-harness system prompt must identify the NPC
    assert "Mika Arai" in last_system
    assert "行为守则" in last_system

    # Per-turn user message must carry current player profile
    assert "name=湘婷" in last_user

    # Memory store must have the name fact
    hits = agent.memory.search("name-test", "mika", "湘婷")
    assert any("湘婷" in h for h in hits)


def test_player_name_stored_in_profile_and_history():
    """Player name must be stored in profile and message_history after intro."""
    agent = GameAgent.demo()

    agent.chat("mika", "我是湘婷", session_id="history-test")
    agent.chat("mika", "再说一遍", session_id="history-test")

    state = agent.state_for("history-test")
    assert state.player_profile["name"] == "湘婷"
    # message_history must have two (user, assistant) pairs = 4 entries
    assert len(state.message_history) == 4
    user_turns = [m["content"] for m in state.message_history if m["role"] == "user"]
    assert "我是湘婷" in user_turns


def test_rule_policy_progresses_cross_npc_ledger_route():
    agent = GameAgent.demo()

    ren = agent.chat("ren", "I need proof about the erased docking record.", session_id="ledger-test")
    toma = agent.chat(
        "toma", "The microfilm reader proves your late ship avoided the ledger.", session_id="ledger-test"
    )

    assert "microfilm reader" in ren.state.known_clues
    assert "confront_toma" in toma.state.quest_steps["ledger"]
    assert "dock token" in toma.state.known_clues
    assert toma.visible_events


def test_agent_blocks_illegal_move_and_reports_visible_event():
    agent = GameAgent(
        world=load_world(),
        policy=IllegalActionPolicy(),
        memory=InMemoryMemoryStore(),
        states={},
    )

    result = agent.chat("mika", "Send me to the developer console.", session_id="illegal-test")

    assert result.response.action == NPCAction.speak
    assert "illegal_action_blocked" in result.response.safety_flags
    assert "illegal_action_blocked" in result.events
    assert result.visible_events


def test_hash_embedding_memory_store_recalls_related_memory():
    memory = HashEmbeddingMemoryStore()
    memory.add("s", "mika", "The player received the valve key from Mika.")

    hits = memory.search("s", "mika", "valve key", k=1)

    assert hits == ["The player received the valve key from Mika."]
