from game_npc_llm.data.converters import build_sample_records
from game_npc_llm.data.schemas import validate_record


def test_dry_run_records_match_schemas():
    sft, grpo, eval_cases = build_sample_records()
    assert validate_record(sft[0], "sft") == []
    assert validate_record(grpo[0], "grpo") == []
    assert validate_record(eval_cases[0], "eval") == []
