from game_npc_llm.data.schemas import (
    NPCAction,
    NPCResponse,
    parse_npc_response,
    repair_npc_response,
    validate_record,
)


def test_npc_response_round_trips_json():
    response = NPCResponse(
        dialogue="Check the microfilm reader before the guild locks the archive.",
        emotion="worried",
        action=NPCAction.reveal_clue,
        target="microfilm reader",
        memory_write=["The player asked about the ledger."],
    )
    parsed, errors = parse_npc_response(response.to_json_text())
    assert errors == []
    assert parsed == response


def test_repair_wraps_plain_text_as_safe_response():
    repaired = repair_npc_response("I can help, but I need a clean action.")
    assert repaired.action == NPCAction.speak
    assert "repaired_output" in repaired.safety_flags


def test_preference_record_schema():
    record = {
        "id": "pref-1",
        "source": "unit",
        "split": "train",
        "prompt": "Player asks about the engine.",
        "chosen": "{\"dialogue\":\"Use the valve key.\"}",
        "rejected": "As an AI, I cannot roleplay.",
        "metadata": {},
    }
    assert validate_record(record, "preference") == []
