from game_npc_llm.data.schemas import validate_record
from scripts.build_product_data import build_kisaragi_samples


def test_kisaragi_samples_validate():
    sft, prefs, eval_cases = build_kisaragi_samples()
    assert len(sft) >= 5
    assert len(prefs) >= 5
    assert len(eval_cases) >= 5
    assert validate_record(sft[0], "sft") == []
    assert validate_record(prefs[0], "preference") == []
    assert validate_record(eval_cases[0], "eval") == []
