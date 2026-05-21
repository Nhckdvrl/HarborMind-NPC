from game_npc_llm.data.schemas import NPCAction
from game_npc_llm.product.agent import GameAgent
from game_npc_llm.product.memory import HashEmbeddingMemoryStore, InMemoryMemoryStore
from game_npc_llm.product.policy import RulePolicyClient
from game_npc_llm.product.world import load_world


class CapturePolicy(RulePolicyClient):
    def __init__(self):
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return super().complete(messages)


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
    policy = CapturePolicy()
    agent = GameAgent(
        world=load_world(),
        policy=policy,
        memory=InMemoryMemoryStore(),
        states={},
    )

    agent.chat("mika", "我是湘婷", session_id="name-test")
    agent.chat("mika", "你不是问过了吗？？", session_id="name-test")

    state = agent.state_for("name-test")
    assert state.player_profile["name"] == "湘婷"
    last_user_prompt = policy.calls[-1][-1]["content"]
    last_system_prompt = policy.calls[-1][0]["content"]
    assert "Known player profile: name: 湘婷" in last_user_prompt
    assert "The player's name is 湘婷. Do not ask for their name again." in last_user_prompt
    assert "Do not invent real-world cities" in last_system_prompt


def test_contextual_guard_answers_known_name_and_location():
    agent = GameAgent.demo()

    intro = agent.chat("mika", "我是湘婷", session_id="guard-test")
    repeat = agent.chat("mika", "你不是问过了吗？？", session_id="guard-test")
    location = agent.chat("mika", "这里是哪儿", session_id="guard-test")

    assert "湘婷" in intro.response.dialogue
    assert "湘婷" in repeat.response.dialogue
    assert "Lantern Pier" in location.response.dialogue
    assert "San Francisco" not in location.response.dialogue


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
