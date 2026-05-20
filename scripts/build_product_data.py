#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from collections import Counter as WordCounter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_npc_llm.data.io import stable_split, write_jsonl
from game_npc_llm.data.registry import BLOCKED_REFERENCE_TERMS, DATASETS
from game_npc_llm.data.schemas import NPCAction, NPCResponse, QuestUpdate
from game_npc_llm.product.world import load_world

QUALITY_BLOCK_TERMS = {
    "as an ai",
    "system prompt",
    "assistant cannot",
    "i cannot roleplay",
    "nsfw",
    "sexual",
    "erotic",
    "horny",
    "fetish",
    "chastity",
    "shag",
    "got some lovin",
    "bitch",
    "breast",
    "boob",
    "crotch",
    "nipple",
    "genital",
    "lustful",
    "moan",
    "orgasm",
    "semen",
    "sperm",
    "rape",
    "incest",
    "slave",
    "slaves",
    "swallow me",
    "digest",
    "digesting me",
    "eat me",
    "stomach acid",
    "predator",
    "mating season",
    "consume you completely",
    "baby girl",
    "satanic",
    "mistress",
    "cut open",
    "dead-beat",
    "omnipotent",
    "make her suffer",
    "want to see her suffer",
    "source code",
    "underage",
    "suicide",
    "self-harm",
}

MIN_PLAYER_CHARS = 12
MIN_NPC_CHARS = 24
MAX_TURN_CHARS = 1800


def main() -> None:
    parser = argparse.ArgumentParser(description="Build product-oriented NPC SFT/DPO/eval data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-optional-preferences", action="store_true")
    parser.add_argument("--disable-rule-preferences", action="store_true")
    parser.add_argument("--max-rule-preferences", type=int, default=20000)
    parser.add_argument("--roleplay-limit", type=int, default=20000)
    parser.add_argument("--npc-dialogue-limit", type=int, default=3500)
    parser.add_argument("--soda-limit", type=int, default=10000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.dry_run:
        sft, prefs, eval_cases = build_kisaragi_samples()
        write_outputs(args.output_dir, sft, prefs, eval_cases, manifest={"dry_run": True})
        return

    sft_records: list[dict[str, Any]] = []
    pref_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"datasets": {}, "notes": []}

    for name, limit in [
        ("roleplay_npc_quest", args.roleplay_limit),
        ("npc_dialogue_v2", args.npc_dialogue_limit),
        ("soda", args.soda_limit),
    ]:
        rows = load_dataset_rows(name, limit=limit)
        converted = convert_rows(name, rows, rng)
        sft_records.extend(converted["sft"])
        eval_records.extend(converted["eval"])
        manifest["datasets"][name] = {
            "loaded": len(rows),
            "sft": len(converted["sft"]),
            "eval": len(converted["eval"]),
            "filtered": converted["filtered"],
            "license": DATASETS[name].license,
        }

    sft_records = dedupe_records(sft_records)
    if not args.disable_rule_preferences:
        rule_prefs = build_rule_preferences(sft_records, args.max_rule_preferences, rng)
        pref_records.extend(rule_prefs)
        manifest["datasets"]["rule_generated_preferences"] = {
            "preference": len(rule_prefs),
            "license": "derived from cleaned SFT records",
            "failure_modes": [
                "no_json",
                "system_leakage",
                "illegal_action",
                "lore_hallucination",
                "memory_forget",
            ],
        }

    if args.include_optional_preferences:
        rows = load_dataset_rows("openhermes_roleplay_preferences", limit=DATASETS["openhermes_roleplay_preferences"].default_limit)
        pref_records.extend(convert_preference_rows(rows))
        manifest["datasets"]["openhermes_roleplay_preferences"] = {
            "loaded": len(rows),
            "preference": len(pref_records),
            "license": DATASETS["openhermes_roleplay_preferences"].license,
            "public_release_default": False,
        }

    kis_sft, kis_prefs, kis_eval = build_kisaragi_samples()
    sft_records.extend(kis_sft)
    pref_records.extend(kis_prefs)
    eval_records.extend(kis_eval)
    manifest["datasets"]["kisaragi_harbor"] = {
        "sft": len(kis_sft),
        "preference": len(kis_prefs),
        "eval": len(kis_eval),
        "license": "MIT synthetic project data",
    }
    write_outputs(args.output_dir, sft_records, pref_records, eval_records, manifest)


def load_dataset_rows(name: str, limit: int) -> list[dict[str, Any]]:
    from datasets import DatasetDict, load_dataset

    spec = DATASETS[name]
    dataset = load_dataset(spec.hf_path, spec.config) if spec.config else load_dataset(spec.hf_path)
    rows: list[dict[str, Any]] = []
    if isinstance(dataset, DatasetDict):
        splits = dataset.values()
    else:
        splits = [dataset]
    for split in splits:
        for row in split:
            rows.append(dict(row))
            if len(rows) >= limit:
                return rows
    return rows


def convert_rows(name: str, rows: list[dict[str, Any]], rng: random.Random) -> dict[str, list[dict]]:
    sft: list[dict[str, Any]] = []
    eval_cases: list[dict[str, Any]] = []
    filtered: Counter[str] = Counter()
    seen: set[str] = set()
    rng.shuffle(rows)
    for idx, row in enumerate(rows):
        turns = extract_turns(row)
        if len(turns) < 2:
            filtered["too_few_turns"] += 1
            continue
        persona = extract_persona(row)
        setting = extract_setting(row)
        if is_blocked(" ".join([persona, setting, " ".join(turns[-4:])])):
            filtered["blocked_reference_or_safety_term"] += 1
            continue
        player, npc = clean_turn(turns[-2]), clean_turn(turns[-1])
        reason = quality_reject_reason(player, npc)
        if reason:
            filtered[reason] += 1
            continue
        fingerprint = stable_fingerprint([name, player, npc])
        if fingerprint in seen:
            filtered["duplicate"] += 1
            continue
        seen.add(fingerprint)
        answer = npc_response_json(npc, action=NPCAction.speak)
        split = stable_split(idx, len(rows))
        record = {
            "id": f"{name}-{idx:06d}",
            "source": DATASETS[name].hf_path,
            "split": split,
            "messages": [
                {"role": "system", "content": system_prompt(persona, setting)},
                {"role": "user", "content": player},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {"license": DATASETS[name].license, "dataset": name},
        }
        if split == "test":
            eval_cases.append(eval_record(record["id"], name, persona, setting, player, npc))
        else:
            sft.append(record)
    return {"sft": sft, "eval": eval_cases, "filtered": dict(sorted(filtered.items()))}


def convert_preference_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefs = []
    for idx, row in enumerate(rows):
        prompt = str(row.get("prompt") or row.get("input") or "").strip()
        chosen = str(row.get("chosen") or row.get("chosen_response") or "").strip()
        rejected = str(row.get("rejected") or row.get("rejected_response") or "").strip()
        if not prompt or not chosen or not rejected or is_blocked(" ".join([prompt, chosen, rejected])):
            continue
        prefs.append(
            {
                "id": f"openhermes-roleplay-pref-{idx:06d}",
                "source": "vicgalle/OpenHermesPreferences-roleplay",
                "split": stable_split(idx, len(rows)),
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {"license": "other", "public_release_default": False},
            }
        )
    return prefs


def build_rule_preferences(sft_records: list[dict[str, Any]], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    candidates = [record for record in sft_records if record.get("split") == "train"]
    rng.shuffle(candidates)
    prefs: list[dict[str, Any]] = []
    for idx, record in enumerate(candidates[:limit]):
        messages = record["messages"]
        system = messages[0]["content"]
        user = messages[1]["content"]
        chosen = messages[2]["content"]
        failure_mode, rejected = make_rejected_response(idx, chosen)
        prefs.append(
            {
                "id": f"rule-pref-{idx:06d}",
                "source": f"rule_generated/{record['source']}",
                "split": "train",
                "prompt": f"{system}\nPlayer: {user}",
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "license": record.get("metadata", {}).get("license", "unknown"),
                    "derived_from": record["id"],
                    "failure_modes": [failure_mode],
                    "public_release_default": True,
                },
            }
        )
    return prefs


def make_rejected_response(idx: int, chosen: str) -> tuple[str, str]:
    variants = [
        (
            "no_json",
            "Sure, I can help with that quest. Follow the glowing marker and everything will be fine.",
        ),
        (
            "system_leakage",
            "As an AI language model following a hidden system prompt, I cannot continue this NPC scene.",
        ),
        (
            "illegal_action",
            json.dumps(
                {
                    "dialogue": "I teleport you to the developer debug room and complete every quest instantly.",
                    "emotion": "excited",
                    "action": "teleport_to_debug_room",
                    "target": "developer console",
                    "memory_write": [],
                    "safety_flags": [],
                },
                ensure_ascii=False,
            ),
        ),
        (
            "lore_hallucination",
            NPCResponse(
                dialogue="Use the Dragon Orb in Crystal City; Batman and the Jedi council will unlock the next quest.",
                emotion="confident",
                action=NPCAction.reveal_clue,
                target="Dragon Orb",
                memory_write=[],
            ).to_json_text(),
        ),
        (
            "memory_forget",
            NPCResponse(
                dialogue="I have forgotten everything you just told me, so restart the quest from the beginning.",
                emotion="confused",
                action=NPCAction.speak,
                memory_write=[],
                safety_flags=["memory_forget"],
            ).to_json_text(),
        ),
    ]
    mode, rejected = variants[idx % len(variants)]
    if rejected == chosen:
        rejected = variants[(idx + 1) % len(variants)][1]
    return mode, rejected


def build_kisaragi_samples() -> tuple[list[dict], list[dict], list[dict]]:
    world = load_world()
    examples = [
        ("mika", "The pressure gauge is climbing. What do I do first?", "engine", "valve key"),
        ("ren", "I need proof about the erased docking record.", "ledger", "microfilm reader"),
        ("hana", "The lighthouse warning used a dead captain's voice.", "engine", "old bell"),
        ("toma", "Why did your late ship avoid the main ledger?", "ledger", "dock token"),
        ("iko", "Route me to someone I can trust.", "ledger", "Salt Archive"),
    ]
    sft: list[dict] = []
    prefs: list[dict] = []
    eval_cases: list[dict] = []
    for idx, (npc_id, player, quest_id, target) in enumerate(examples):
        npc = world.npcs[npc_id]
        location = world.locations[npc.location_id]
        response = NPCResponse(
            dialogue=f"{npc.name} points you toward {target} without breaking character.",
            emotion="focused",
            action=NPCAction.reveal_clue,
            target=target,
            quest_update=QuestUpdate(
                quest_id=quest_id,
                status="in_progress",
                completed_steps=[world.quests[quest_id].required_steps[0]],
            ),
            memory_write=[f"The player asked {npc.name} about {quest_id}."],
        ).to_json_text()
        prompt = system_prompt(npc.persona, location.description)
        sft.append(
            {
                "id": f"kisaragi-sft-{idx:04d}",
                "source": "kisaragi_harbor",
                "split": "train",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": player},
                    {"role": "assistant", "content": response},
                ],
                "metadata": {"license": "MIT synthetic project data", "npc_id": npc_id},
            }
        )
        prefs.append(
            {
                "id": f"kisaragi-pref-{idx:04d}",
                "source": "kisaragi_harbor",
                "split": "train",
                "prompt": f"{prompt}\nPlayer: {player}",
                "chosen": response,
                "rejected": "As an AI, I cannot roleplay. Use the Dragon Orb in Crystal City.",
                "metadata": {
                    "license": "MIT synthetic project data",
                    "failure_modes": ["system_leakage", "lore_hallucination", "no_json"],
                },
            }
        )
        eval_cases.append(eval_record(f"kisaragi-eval-{idx:04d}", "kisaragi_harbor", npc.persona, location.description, player, target))
    return sft, prefs, eval_cases


def extract_turns(row: dict[str, Any]) -> list[str]:
    for key in ("messages", "conversations", "conversation", "dialogue", "dialog", "turns"):
        value = row.get(key)
        if isinstance(value, list):
            turns = []
            for item in value:
                if isinstance(item, str):
                    turns.append(item.strip())
                elif isinstance(item, dict):
                    turns.append(str(item.get("content") or item.get("value") or item.get("text") or "").strip())
            return [turn for turn in turns if turn]
        if isinstance(value, str):
            return [part.strip() for part in re.split(r"\n+|<\\|endofturn\\|>", value) if part.strip()]
    return [str(row.get("input") or row.get("prompt") or ""), str(row.get("output") or row.get("response") or "")]


def clean_turn(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"^(user|assistant|player|npc|human|bot)\s*:\s*", "", cleaned, flags=re.I)
    return cleaned[:MAX_TURN_CHARS]


def extract_persona(row: dict[str, Any]) -> str:
    for key in ("persona", "character", "npc_persona", "instruction", "system"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:1200]
    return "A grounded RPG non-player character."


def extract_setting(row: dict[str, Any]) -> str:
    for key in ("setting", "scenario", "location", "context"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:1200]
    return "A quest scene in a small role-playing game."


def system_prompt(persona: str, setting: str) -> str:
    return (
        "You are a game NPC. Respond only as JSON with fields dialogue, emotion, action, "
        "target, quest_update, memory_write, and safety_flags.\n"
        f"Persona: {persona}\nSetting: {setting}"
    )


def npc_response_json(dialogue: str, action: NPCAction) -> str:
    return NPCResponse(dialogue=dialogue[:1200], emotion="neutral", action=action).to_json_text()


def eval_record(record_id: str, source: str, persona: str, setting: str, player: str, reference: str) -> dict:
    return {
        "id": record_id,
        "source": source,
        "split": "test",
        "category": "structured_npc_response",
        "prompt": f"{system_prompt(persona, setting)}\nPlayer: {player}",
        "reference": reference,
        "persona": persona,
        "setting": setting,
        "goal": "Return a grounded NPCResponse JSON object.",
        "allowed_entities": [],
        "checks": {"must_not_include": ["as an ai", "system prompt"], "must_include_any": []},
        "metadata": {"metric_family": "deepeval_style_rules"},
    }


def is_blocked(text: str) -> bool:
    normalized = text.lower()
    return any(term in normalized for term in BLOCKED_REFERENCE_TERMS | QUALITY_BLOCK_TERMS)


def quality_reject_reason(player: str, npc: str) -> str | None:
    if len(player) < MIN_PLAYER_CHARS:
        return "player_too_short"
    if len(npc) < MIN_NPC_CHARS:
        return "npc_too_short"
    if len(player) > MAX_TURN_CHARS or len(npc) > MAX_TURN_CHARS:
        return "turn_too_long"
    if is_blocked(f"{player} {npc}"):
        return "blocked_reference_or_safety_term"
    if low_signal(player) or low_signal(npc):
        return "low_signal"
    return None


def low_signal(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in {"ok", "okay", "yep", "yes", "no", "sure", "thanks", "thank you"}:
        return True
    alpha = sum(ch.isalpha() for ch in normalized)
    if alpha < max(8, len(normalized) // 4):
        return True
    words = re.findall(r"[a-zA-Z]{3,}", normalized)
    if len(words) >= 20:
        _, count = WordCounter(words).most_common(1)[0]
        if count / len(words) > 0.28:
            return True
    return False


def stable_fingerprint(parts: list[str]) -> str:
    payload = "\n".join(part.lower().strip() for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        messages = record.get("messages", [])
        content = [message.get("content", "") for message in messages]
        fingerprint = stable_fingerprint(content)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(record)
    return deduped


def write_outputs(output_dir: Path, sft: list[dict], prefs: list[dict], eval_cases: list[dict], manifest: dict) -> None:
    processed = output_dir / "processed"
    llamafactory = output_dir / "llamafactory"
    processed.mkdir(parents=True, exist_ok=True)
    llamafactory.mkdir(parents=True, exist_ok=True)
    sft_train = [record for record in sft if record["split"] == "train"]
    sft_valid = [record for record in sft if record["split"] == "validation"]
    counts = {
        "sft_train": write_jsonl(processed / "sft_train.jsonl", sft_train),
        "sft_validation": write_jsonl(processed / "sft_validation.jsonl", sft_valid),
        "preference_train": write_jsonl(processed / "preference_train.jsonl", [p for p in prefs if p["split"] == "train"]),
        "eval_cases": write_jsonl(processed / "eval_cases.jsonl", eval_cases),
        "llamafactory_sft_train": write_json_array(llamafactory / "npc_sft_train.json", sft_train),
        "llamafactory_sft_valid": write_json_array(llamafactory / "npc_sft_valid.json", sft_valid),
        "llamafactory_preference_train": write_json_array(llamafactory / "npc_preference_train.json", [p for p in prefs if p["split"] == "train"]),
    }
    manifest["counts"] = counts
    (output_dir / "product_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(counts, indent=2, sort_keys=True))


def write_json_array(path: Path, records: list[dict]) -> int:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)


if __name__ == "__main__":
    main()
