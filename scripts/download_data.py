#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_npc_llm.data.converters import (
    build_sample_records,
    convert_light_dialog,
    convert_light_quests,
    convert_npc_dialogue,
    convert_npc_quest_dialogue,
)
from game_npc_llm.data.io import write_jsonl


LIGHT_TASKS = {
    "light_dialog": ["dap-exp/light_dialog"],
    "light_dialog_wild": ["facebook/light", "light_dialog_wild"],
    "light_quests": ["facebook/light", "light_quests"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert public NPC datasets.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Write tiny sample files only.")
    parser.add_argument("--skip-light", action="store_true")
    parser.add_argument("--skip-npc-dialogue", action="store_true")
    parser.add_argument("--skip-npc-quest-dialogue", action="store_true")
    args = parser.parse_args()

    processed = args.output_dir / "processed"
    raw = args.output_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        sft, grpo, eval_cases = build_sample_records()
        write_outputs(processed, sft, grpo, eval_cases)
        dump_manifest(
            args.output_dir / "raw_manifest.json",
            {
                "generated_at": now_iso(),
                "dry_run": True,
                "sources": {"dry_run": {"records": len(sft) + len(grpo) + len(eval_cases)}},
            },
        )
        print("Dry run complete. Wrote sample processed JSONL files.")
        return

    sft_records: list[dict[str, Any]] = []
    grpo_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []

    if not args.skip_npc_dialogue:
        npc_rows = flatten_dataset(load_hf_dataset("chimbiwide/NPC-Dialogue_v2", config="dialogue"))
        npc_sft, npc_eval = convert_npc_dialogue(npc_rows, seed=args.seed)
        sft_records.extend(npc_sft)
        eval_records.extend(npc_eval)
        dump_manifest(raw / "npc_dialogue_v2_manifest.json", {"records": len(npc_rows)})

    if not args.skip_npc_quest_dialogue:
        npc_quest_rows = flatten_dataset(load_hf_dataset("chimbiwide/NPC-Quest-Dialogue"))
        npc_quest_grpo, npc_quest_eval = convert_npc_quest_dialogue(npc_quest_rows, seed=args.seed)
        grpo_records.extend(npc_quest_grpo)
        eval_records.extend(npc_quest_eval)
        dump_manifest(raw / "npc_quest_dialogue_manifest.json", {"records": len(npc_quest_rows)})

    if not args.skip_light:
        for task_name, hf_args in LIGHT_TASKS.items():
            rows = try_load_hf_light(hf_args)
            if rows is None:
                rows = try_load_parlai_task(task_name, raw / "parlai")
            if rows is None:
                print(f"WARNING: could not load {task_name}; install ParlAI or provide HF mirror.")
                continue
            if task_name == "light_quests":
                quest_grpo, quest_eval = convert_light_quests(rows, task_name)
                grpo_records.extend(quest_grpo)
                eval_records.extend(quest_eval)
            else:
                sft_records.extend(convert_light_dialog(rows, task_name))
            dump_manifest(raw / f"{task_name}_manifest.json", {"records": len(rows)})

    write_outputs(processed, sft_records, grpo_records, eval_records)
    dump_manifest(
        args.output_dir / "raw_manifest.json",
        {
            "generated_at": now_iso(),
            "seed": args.seed,
            "sources": {
                "sft_records": len(sft_records),
                "grpo_records": len(grpo_records),
                "eval_records": len(eval_records),
            },
            "notes": [
                "NPC-Dialogue_v2 records with obvious copyrighted IP or celebrity references are filtered during conversion.",
                "LIGHT dialogue uses the dap-exp/light_dialog Hugging Face mirror when available.",
                "LIGHT-WILD and LIGHT-Quests fall back to ParlAI tasks when installed.",
                "NPC-Quest-Dialogue is used as an Apache-2.0 fallback/supplement for GRPO quest prompts.",
            ],
        },
    )


def flatten_dataset(dataset: Any) -> list[dict[str, Any]]:
    from datasets import DatasetDict

    rows: list[dict[str, Any]] = []
    if isinstance(dataset, DatasetDict):
        for split, split_ds in dataset.items():
            for row in split_ds:
                item = dict(row)
                item.setdefault("split", split)
                rows.append(item)
    else:
        for row in dataset:
            rows.append(dict(row))
    return rows


def try_load_hf_light(hf_args: list[str]) -> list[dict[str, Any]] | None:
    try:
        return flatten_dataset(load_hf_dataset(*hf_args))
    except Exception as exc:
        print(f"HF load failed for {hf_args}: {exc}")
        return None


def load_hf_dataset(path: str, config: str | None = None) -> Any:
    from datasets import load_dataset

    try:
        return load_dataset(path, config) if config else load_dataset(path)
    except ValueError as exc:
        message = str(exc)
        if "Config name is missing" in message and config is None:
            if path == "chimbiwide/NPC-Dialogue_v2":
                return load_dataset(path, "dialogue")
        raise


def try_load_parlai_task(task_name: str, datadir: Path) -> list[dict[str, Any]] | None:
    try:
        import parlai  # noqa: F401
    except Exception:
        return None
    cmd = [
        sys.executable,
        "-m",
        "parlai.scripts.display_data",
        "--task",
        task_name,
        "--datapath",
        str(datadir),
        "--num-examples",
        "100000000",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"ParlAI load failed for {task_name}: {exc.stderr[-1000:]}")
        return None
    return parse_parlai_display(result.stdout, task_name)


def parse_parlai_display(text: str, task_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] = {"source": task_name, "dialogue": []}
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if clean.startswith("[") and "episode" in clean.lower():
            if current.get("dialogue"):
                rows.append(current)
            current = {"source": task_name, "dialogue": []}
            continue
        if clean.startswith("text:"):
            current["dialogue"].append(clean.removeprefix("text:").strip())
        elif clean.startswith("labels:") or clean.startswith("eval_labels:"):
            current["dialogue"].append(clean.split(":", 1)[1].strip())
        elif ":" in clean:
            key, value = clean.split(":", 1)
            current[key.strip()] = value.strip()
    if current.get("dialogue"):
        rows.append(current)
    return rows


def dump_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_outputs(processed: Path, sft: list[dict], grpo: list[dict], eval_cases: list[dict]) -> None:
    llamafactory = processed.parent / "llamafactory"
    eval_dir = processed.parent / "eval"
    llamafactory.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    sft_train = [record for record in sft if record["split"] == "train"]
    sft_validation = [record for record in sft if record["split"] == "validation"]
    sft_test = [record for record in sft if record["split"] == "test"]
    counts = {
        "sft_train": write_jsonl(processed / "sft_train.jsonl", sft_train),
        "sft_validation": write_jsonl(processed / "sft_validation.jsonl", sft_validation),
        "sft_test": write_jsonl(processed / "sft_test.jsonl", sft_test),
        "grpo_prompts": write_jsonl(processed / "grpo_prompts.jsonl", grpo),
        "eval_cases": write_jsonl(processed / "eval_cases.jsonl", eval_cases),
        "llamafactory_train": write_json_array(llamafactory / "npc_sft_train.json", sft_train),
        "llamafactory_valid": write_json_array(llamafactory / "npc_sft_valid.json", sft_validation),
        "eval_cases_copy": write_jsonl(eval_dir / "eval_cases.jsonl", eval_cases),
    }
    print(json.dumps(counts, indent=2, sort_keys=True))


def write_json_array(path: Path, records: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)


if __name__ == "__main__":
    main()
