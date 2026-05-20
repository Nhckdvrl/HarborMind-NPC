from __future__ import annotations

import json
import statistics
from pathlib import Path

from game_npc_llm.eval.metrics import evaluate_generation

try:
    from lm_eval.api.task import Task
except Exception:
    Task = object  # type: ignore[misc,assignment]


class GameNPCBench(Task):  # type: ignore[misc]
    VERSION = 0

    def __init__(self, data_path: str = "data/processed/eval_cases.jsonl", **kwargs):
        if Task is not object:
            super().__init__(**kwargs)
        self.data_path = Path(data_path)
        self._docs = self._load_docs()

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return [doc for doc in self._docs if doc.get("split") in {"validation", "test"}]

    def test_docs(self):
        return [doc for doc in self._docs if doc.get("split") == "test"]

    def doc_to_text(self, doc):
        return doc["prompt"]

    def doc_to_target(self, doc):
        return doc.get("reference", "")

    def construct_requests(self, doc, ctx):
        return self._model_generate(ctx)

    def process_results(self, doc, results):
        generation = results[0] if isinstance(results, (list, tuple)) else str(results)
        return evaluate_generation(doc, generation)

    def aggregation(self):
        return {
            "json_validity": statistics.mean,
            "action_validity": statistics.mean,
            "role_adherence": statistics.mean,
            "quest_progression": statistics.mean,
            "system_leakage_rate": statistics.mean,
            "memory_write_rate": statistics.mean,
            "average_latency": statistics.mean,
            "tokens_per_second": statistics.mean,
        }

    def higher_is_better(self):
        return {
            "json_validity": True,
            "action_validity": True,
            "role_adherence": True,
            "quest_progression": True,
            "system_leakage_rate": False,
            "memory_write_rate": True,
            "average_latency": False,
            "tokens_per_second": True,
        }

    def _load_docs(self):
        if not self.data_path.exists():
            return []
        docs = []
        with self.data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    docs.append(json.loads(line))
        return docs

    def _model_generate(self, ctx):
        if hasattr(self, "rf"):
            return self.rf.greedy_until(ctx, {"until": ["\n\n"], "max_gen_toks": 256})
        return ""
