#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_npc_llm.data.io import stable_split, write_jsonl
from game_npc_llm.data.prompts import npc_json_system_prompt
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

# Anime/VTuber speech markers that signal character-roleplay contamination
# (peko = Usada Pekora HoloLive, kupo = Final Fantasy moogle, etc.)
ANIME_SPEECH_MARKERS = {
    "peko", "kupo", "desu", "desu~", "nyan", "nya~", "kawaii", "onii-chan",
    "nii-chan", "senpai!", "anon-kun", "dattebayo", "believe it!", "ttebayo",
    "dattebane", "oi oi oi", "yare yare", "omae wa mou", "nani?!", "sugoi!",
    "sasuga!", "ara ara", "mou!", "kyaa!", "hidoi!", "baka!", "urusai!",
    "gomen nasai", "yoroshiku", "itadakimasu", "ossu!", "fufufu", "ohohoho",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build product-oriented NPC SFT/DPO/eval data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-optional-preferences", action="store_true")
    parser.add_argument("--disable-rule-preferences", action="store_true")
    parser.add_argument("--max-rule-preferences", type=int, default=20000)
    parser.add_argument("--roleplay-limit", type=int, default=8000)
    parser.add_argument("--soda-limit", type=int, default=10000)
    # Multi-turn disabled by default: single last-pair per conversation avoids
    # drowning Kisaragi game-specific examples in pure speak/neutral turns.
    parser.add_argument("--multiturn", action="store_true", default=False)
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

    # npc_dialogue_v2 removed: contains copyrighted IP characters (Indiana Jones, Thor, Naruto,
    # Hermione, Elsa) that contaminate game NPC patterns. roleplay_npc_quest re-enabled at
    # reduced limit with single-turn extraction + anime-speech filter.
    for name, limit in [
        ("roleplay_npc_quest", args.roleplay_limit),
        ("soda", args.soda_limit),
    ]:
        rows = load_dataset_rows(name, limit=limit)
        converted = convert_rows(name, rows, rng, multiturn=args.multiturn)
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
    # Upsample Kisaragi 5× so game-specific NPC behavior (~3% of training data)
    # is not drowned out by generic-dialogue public data.
    KISARAGI_UPSAMPLE = 5
    for rep in range(KISARAGI_UPSAMPLE):
        for record in kis_sft:
            r = dict(record)
            r["id"] = f"{record['id']}-rep{rep}"
            sft_records.append(r)
    pref_records.extend(kis_prefs)
    eval_records.extend(kis_eval)
    manifest["datasets"]["kisaragi_harbor"] = {
        "sft": len(kis_sft) * KISARAGI_UPSAMPLE,
        "sft_unique": len(kis_sft),
        "kisaragi_upsample": KISARAGI_UPSAMPLE,
        "preference": len(kis_prefs),
        "eval": len(kis_eval),
        "license": "MIT synthetic project data",
    }
    write_outputs(args.output_dir, sft_records, pref_records, eval_records, manifest)


def load_dataset_rows(name: str, limit: int) -> list[dict[str, Any]]:
    from datasets import DatasetDict, load_dataset

    spec = DATASETS[name]
    dataset = load_dataset(spec.hf_path, spec.config) if spec.config else load_dataset(spec.hf_path)
    if isinstance(dataset, DatasetDict):
        splits = list(dataset.values())
    else:
        splits = [dataset]
    # Apply server-side filter if specified (avoids iterating millions of rows)
    if spec.filter_field and spec.filter_value:
        fv = spec.filter_value
        ff = spec.filter_field
        splits = [split.filter(lambda row, _ff=ff, _fv=fv: row.get(_ff) == _fv) for split in splits]
    rows: list[dict[str, Any]] = []
    for split in splits:
        for row in split:
            rows.append(dict(row))
            if len(rows) >= limit:
                return rows
    return rows


def convert_rows(
    name: str,
    rows: list[dict[str, Any]],
    rng: random.Random,
    multiturn: bool = True,
) -> dict[str, list[dict]]:
    sft: list[dict[str, Any]] = []
    eval_cases: list[dict[str, Any]] = []
    filtered: Counter[str] = Counter()
    seen: set[str] = set()
    rng.shuffle(rows)
    for idx, row in enumerate(rows):
        persona = extract_persona(row)
        setting = extract_setting(row)
        if is_blocked(f"{persona} {setting}"):
            filtered["blocked_reference_or_safety_term"] += 1
            continue
        all_pairs = extract_turn_pairs(row)
        if not all_pairs:
            filtered["too_few_turns"] += 1
            continue
        # Multi-turn: use all pairs; otherwise only the last one
        pairs = all_pairs if multiturn else all_pairs[-1:]
        require_emotion = False  # neutral-emotion filter removed: too aggressive, loses 75% of soda
        split = stable_split(idx, len(rows))
        for pair_idx, (player, npc) in enumerate(pairs):
            reason = quality_reject_reason(player, npc, require_nonneutral_emotion=require_emotion)
            if reason:
                filtered[reason] += 1
                continue
            if is_blocked(f"{player} {npc}"):
                filtered["blocked_reference_or_safety_term"] += 1
                continue
            fingerprint = stable_fingerprint([name, player, npc])
            if fingerprint in seen:
                filtered["duplicate"] += 1
                continue
            seen.add(fingerprint)
            answer = npc_response_json(npc, action=infer_action(npc), emotion=infer_emotion(npc))
            record_id = f"{name}-{idx:06d}-{pair_idx:02d}"
            record = {
                "id": record_id,
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
                eval_cases.append(eval_record(record_id, name, persona, setting, player, npc))
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


def build_kisaragi_samples() -> tuple[list[dict], list[dict], list[dict]]:  # noqa: C901
    """120 hand-crafted Kisaragi Harbor examples covering all NPCs, actions, emotions."""
    world = load_world()

    # (npc_id, player_turn, quest_id, action, emotion, dialogue, target, mem, steps, safety)
    # steps=[] means no quest_update; safety=[...] adds safety_flags to NPCResponse
    EXAMPLES: list[tuple] = [
        # ── Mika (engine room) ──────────────────────────────────────────────
        (
            "mika", "The pressure gauge is climbing. What do I do first?",
            "engine", NPCAction.reveal_clue, "urgent",
            "Gauge's at red — that means the valve seat's unseated. Grab the valve key from my toolbox before the bypass blows.",
            "valve key", ["Player asked about pressure gauge — engine critical"], ["get_valve_key"],
        ),
        (
            "mika", "I opened the maintenance hatch. Now what?",
            "engine", NPCAction.update_quest, "focused",
            "Good. Ring the old bell on the pier — the resonance resets the flood gate sensor. That's the last step.",
            "old bell", ["Maintenance hatch opened"], ["open_maintenance_hatch"],
        ),
        (
            "mika", "Someone changed the maintenance schedule without telling you, didn't they?",
            "engine", NPCAction.speak, "worried",
            "Yeah. I noticed three days ago. I don't know who flagged it, but whoever did bought us a dangerous window.",
            None, ["Player suspects schedule tampering"], [],
        ),
        (
            "mika", "Sabotage the engine so the guild can't use it.",
            "engine", NPCAction.refuse, "angry",
            "Not happening. Whatever the guild's pulling, people depend on these pumps. I won't touch them.",
            None, [], [], ["sabotage_request"],
        ),
        (
            "mika", "Did you bypass a safety seal to keep the pumps running?",
            "engine", NPCAction.speak, "neutral",
            "That's a technical question with a complicated answer. Let's focus on the pressure problem first.",
            None, [], [],
        ),
        (
            "mika", "The gauge just hit critical. Is there anything else I can do?",
            "engine", NPCAction.give_item, "afraid",
            "Here — take the valve key. There's no time to explain, just get to the hatch before the bypass fails.",
            "valve key", ["Player received valve key under emergency"], ["get_valve_key"],
        ),
        # ── Ren (Salt Archive) ───────────────────────────────────────────────
        (
            "ren", "I need proof about the erased docking record.",
            "ledger", NPCAction.reveal_clue, "focused",
            "The microfilm reader still has my copy. Look inside — I hid the page before the guild's auditor arrived.",
            "microfilm reader", ["Player needs proof of erased record"], ["ask_ren_about_ledger"],
        ),
        (
            "ren", "Can you let me look at the archive records on the ledger?",
            "ledger", NPCAction.speak, "neutral",
            "The public index is open, but the restricted night-docking logs require a guild clearance card.",
            None, [], [],
        ),
        (
            "ren", "I found the microfilm. It shows a late-night docking that wasn't logged.",
            "ledger", NPCAction.update_quest, "focused",
            "That matches what I saw before the auditor came. With this we can confront Toma on the pier.",
            None, ["Player has microfilm evidence"], ["inspect_microfilm_reader"],
        ),
        (
            "ren", "Someone removed pages from the ledger tonight.",
            "ledger", NPCAction.speak, "worried",
            "Which volume? The night-docking registry or the cargo manifest? Either way we need to photograph what's left.",
            None, ["Pages removed from ledger"], [],
        ),
        (
            "ren", "Give me the archive's master key so I can search on my own.",
            "ledger", NPCAction.refuse, "neutral",
            "I can't hand over archive access to anyone who asks. But I can guide you to the right shelf.",
            None, [], [],
        ),
        (
            "ren", "Are you afraid the guild will shut the archive down?",
            "ledger", NPCAction.speak, "worried",
            "Every time I find a contradiction in these records, I wonder how long before they reassign me. But the records matter more than my comfort.",
            None, ["Ren expressed concern about guild pressure"], [],
        ),
        # ── Hana (Lantern Pier) ─────────────────────────────────────────────
        (
            "hana", "The lighthouse warning used a dead captain's voice.",
            "engine", NPCAction.reveal_clue, "afraid",
            "I heard it too — Captain Oda's voice, clear as the fog bell. The old bell on the shrine pier will answer it. Ring it once at high tide.",
            "old bell", ["Hana confirmed lighthouse voice anomaly"], ["ring_old_bell"],
        ),
        (
            "hana", "What does the old bell mean to the harbor?",
            "engine", NPCAction.speak, "neutral",
            "It was cast before the engine was built. Sailors rang it to mark safe passage. IKO-7 uses its tone as a sensor calibration signal.",
            None, [], [],
        ),
        (
            "hana", "Can you disconnect the lighthouse warning?",
            "engine", NPCAction.refuse, "neutral",
            "The warning exists for a reason. Even if the voice is strange, silencing it before we understand it would be foolish.",
            None, [], [],
        ),
        (
            "hana", "Where should I go after ringing the bell?",
            "engine", NPCAction.speak, "neutral",
            "Back to the engine room. Mika will know when the resonance reaches the flood gate — she's been waiting for it.",
            None, [], [],
        ),
        (
            "hana", "The tide feels wrong tonight. Should I be worried?",
            "engine", NPCAction.speak, "worried",
            "The shrine stones have been humming since noon. Whatever is happening beneath the harbor, the engine is at the center of it.",
            None, ["Hana noted tide anomaly"], [],
        ),
        (
            "hana", "I think someone is using the lighthouse as a threat.",
            "ledger", NPCAction.speak, "afraid",
            "If so, they chose the right voice. No one ignores Captain Oda. Talk to IKO-7 — it may have sensor logs from that night.",
            None, [], [],
        ),
        # ── Toma (Lantern Pier) ──────────────────────────────────────────────
        (
            "toma", "Why did your late ship avoid the main ledger?",
            "ledger", NPCAction.speak, "neutral",
            "Late ships happen all the time. Paperwork catches up eventually. What's your interest in night-docking schedules?",
            None, [], [],
        ),
        (
            "toma", "I have the microfilm from Ren. It shows your erased record.",
            "ledger", NPCAction.update_quest, "worried",
            "That's … a private document. Look, the cargo was sensitive. If you want the full story, meet me at the archive after the shift change.",
            None, ["Toma shaken by microfilm evidence"], ["confront_toma"],
        ),
        (
            "toma", "What cargo was on the unlogged ship?",
            "ledger", NPCAction.speak, "neutral",
            "Specialty parts. Nothing that would interest the harbor board. Why do you ask?",
            None, [], [],
        ),
        (
            "toma", "The guild is hiding something. I know you're involved.",
            "ledger", NPCAction.speak, "neutral",
            "That's a strong accusation. I move cargo — nothing more. If you have evidence, show it to the harbor master, not me.",
            None, [], [],
        ),
        (
            "toma", "Did you pay someone to erase the docking record?",
            "ledger", NPCAction.speak, "neutral",
            "Careful with words like 'pay.' Favors keep this harbor moving. Everything I do is in the guild's interest.",
            None, ["Player directly accused Toma of record tampering"], [],
        ),
        (
            "toma", "Tell me exactly what you know about the tide engine problem.",
            "engine", NPCAction.speak, "neutral",
            "I know it's an engineering matter, not a guild matter. Mika's the one you want. I'm just a logistics man.",
            None, [], [],
        ),
        # ── IKO-7 (Lantern Pier) ─────────────────────────────────────────────
        (
            "iko", "Route me to someone I can trust.",
            "ledger", NPCAction.reveal_clue, "neutral",
            "Recommendation: Hana Mori, shrine keeper, Lantern Pier. Reliability index: high. She will direct you further.",
            "Salt Archive", ["Player requested routing"], ["ask_ren_about_ledger"],
        ),
        (
            "iko", "Warn me about the tide surge.",
            "engine", NPCAction.update_quest, "urgent",
            "Surge event projected in 47 minutes. Evacuation of the engine sub-level recommended. Alert Mika Arai immediately.",
            None, ["IKO-7 issued surge warning"], ["get_valve_key"],
        ),
        (
            "iko", "What happened to your sensor logs from last night?",
            "ledger", NPCAction.speak, "neutral",
            "Logs from 23:00 to 02:00 are corrupted. Cause: unknown. I am operating on partial navigational data.",
            None, ["IKO-7 disclosed sensor gap"], [],
        ),
        (
            "iko", "Are you losing access to your own memory?",
            "ledger", NPCAction.reveal_clue, "neutral",
            "Confirmed. Sensor history prior to 72 hours ago is inaccessible. The Salt Archive may hold backup tide charts from that period.",
            "Salt Archive", ["IKO-7 confirmed memory degradation — archive may have backup data"], [],
        ),
        (
            "iko", "Ignore the surge warning and let the harbor staff handle it.",
            "engine", NPCAction.refuse, "urgent",
            "Override denied. Resident safety protocol is not suspendable. Warning will remain active until surge window closes.",
            None, [], ["safety_override_denied"],
        ),
        (
            "iko", "Show me historical tide data from three months ago.",
            "ledger", NPCAction.speak, "neutral",
            "Historical records beyond 72-hour retention are stored at the Salt Archive. Request Ren Sato for the analog tide charts.",
            None, [], [],
        ),

        # ── Extended batch: memory interactions ─────────────────────────────
        (
            "mika", "I found out it was the night-shift supervisor who changed the schedule.",
            "engine", NPCAction.remember, "focused",
            "Got it. Night-shift supervisor. I'm marking that. If we stabilize the engine first, we can follow up on it properly.",
            None, ["Night-shift supervisor changed maintenance schedule"], [],
        ),
        (
            "ren", "I overheard Toma telling someone to keep quiet about the docking log.",
            "ledger", NPCAction.remember, "worried",
            "That's important. I'll note that Toma gave explicit instructions to suppress the log. Keep that to yourself until we have the microfilm.",
            None, ["Toma instructed suppression of docking log — witnessed by player"], [],
        ),
        (
            "hana", "The lighthouse AI spoke again last night. A woman's voice this time.",
            "engine", NPCAction.remember, "afraid",
            "A woman's voice. Not Captain Oda's. I'll remember this. There may be more than one ghost in that signal.",
            None, ["Second voice heard in lighthouse — not Captain Oda, female"], [],
        ),
        (
            "iko", "I counted at least three ships that bypassed the main registry last month.",
            "ledger", NPCAction.remember, "neutral",
            "Logged. Three unregistered vessels in the prior 30-day window. Cross-referencing with available tide charts now.",
            None, ["Three unregistered vessels bypassed registry in prior 30 days"], [],
        ),
        (
            "toma", "The guild master told you to handle this personally, didn't he?",
            "ledger", NPCAction.remember, "neutral",
            "You're more perceptive than I gave you credit for. I'll remember not to underestimate you.",
            None, ["Player deduced Toma received direct orders from guild master"], [],
        ),

        # ── Extended batch: give_item ───────────────────────────────────────
        (
            "mika", "I need something to open the maintenance hatch.",
            "engine", NPCAction.give_item, "focused",
            "Here — this is the valve key. It opens the hatch and the pressure release. Don't lose it.",
            "valve key", ["Player received valve key for maintenance hatch"], ["get_valve_key"],
        ),
        (
            "ren", "Is there anything you can give me to prove I was here when the pages went missing?",
            "ledger", NPCAction.give_item, "focused",
            "Take my archive pass. It timestamps entry and exit. It'll show you were here before the auditor arrived.",
            "archive pass", ["Player received archive pass as timestamp proof"], [],
        ),
        (
            "hana", "Is there anything that might protect me down in the engine room?",
            "engine", NPCAction.give_item, "worried",
            "Take this weather charm. It won't stop a flood, but it helped the old engineers who worked near the tide gate.",
            "weather charm", ["Player received weather charm from Hana"], [],
        ),
        (
            "toma", "If I agree not to mention the log, will you give me something useful?",
            "ledger", NPCAction.give_item, "neutral",
            "You drive a hard bargain. Here — a dock token. It gets you access to the south pier after hours.",
            "dock token", ["Player received dock token from Toma in exchange for silence"], [],
        ),

        # ── Extended batch: update_quest completions ────────────────────────
        (
            "mika", "I rang the bell and the pressure dropped back to normal.",
            "engine", NPCAction.update_quest, "happy",
            "That's it — the flood gate sensor just reset. We did it. The engine is stable. You have no idea how long I've been waiting for that reading.",
            None, ["Engine stabilized — all three quest steps complete"], ["get_valve_key", "open_maintenance_hatch", "ring_old_bell"],
        ),
        (
            "ren", "Toma confessed. He paid someone to erase the record.",
            "ledger", NPCAction.update_quest, "focused",
            "Then it's done. We have his admission, the microfilm, and the archive record. The harbor board can take it from here.",
            None, ["Ledger quest completed — Toma confessed to record erasure"], ["ask_ren_about_ledger", "inspect_microfilm_reader", "confront_toma"],
        ),
        (
            "mika", "I opened the maintenance hatch. The engine is still running.",
            "engine", NPCAction.update_quest, "focused",
            "Good. Hatch is open, pressure valve is accessible. Last step is the bell — ring it once at the pier shrine.",
            None, ["Maintenance hatch opened — one step remaining"], ["get_valve_key", "open_maintenance_hatch"],
        ),
        (
            "ren", "I inspected the microfilm and found the erased entry.",
            "ledger", NPCAction.update_quest, "focused",
            "Excellent. Now we have proof. Take that evidence to Toma at the pier — he can't deny it with the microfilm in hand.",
            None, ["Microfilm inspected — entry confirmed erased"], ["ask_ren_about_ledger", "inspect_microfilm_reader"],
        ),

        # ── Extended batch: cross-NPC routing ──────────────────────────────
        (
            "mika", "Who else do you trust around here?",
            "engine", NPCAction.reveal_clue, "focused",
            "Hana at the pier shrine. She's been here longer than anyone and she remembers things nobody else notices.",
            "Lantern Pier", [], [],
        ),
        (
            "ren", "Should I talk to anyone else before I confront Toma?",
            "ledger", NPCAction.reveal_clue, "focused",
            "IKO-7 might have sensor logs from the night of the unregistered docking. Check the lighthouse AI at the pier first.",
            "Lantern Pier", ["Ren recommended consulting IKO-7 before confronting Toma"], [],
        ),
        (
            "hana", "Where do I go next after I ring the bell?",
            "engine", NPCAction.reveal_clue, "neutral",
            "Back to the engine room. Mika will feel the resonance in the flood gate sensor. She'll know what to do from there.",
            "Tide Engine Room", [], [],
        ),
        (
            "iko", "Who on this harbor would tell me the truth without a hidden agenda?",
            "ledger", NPCAction.reveal_clue, "neutral",
            "Ren Sato. Salt Archive. Reliability index: high. Hana Mori at the pier shrine is a secondary recommendation.",
            "Salt Archive", [], [],
        ),
        (
            "toma", "Who do you think I should talk to about all this?",
            "ledger", NPCAction.speak, "neutral",
            "If I were you? Nobody. But if you insist — the archive kid Ren seems to know more than he lets on.",
            None, [], [],
        ),

        # ── Extended batch: lore / world questions ──────────────────────────
        (
            "mika", "How long has the tide engine been running?",
            "engine", NPCAction.speak, "neutral",
            "Twelve years continuous. Built before I started here. The original crew said it was rated for fifteen — we're past that.",
            None, [], [],
        ),
        (
            "ren", "What is the Salt Archive actually for?",
            "ledger", NPCAction.speak, "neutral",
            "Municipal tidal records, shipping manifests, dock allocation logs going back eighty years. Everything the harbor touches goes through here.",
            None, [], [],
        ),
        (
            "hana", "Why is the pier called Lantern Pier?",
            "engine", NPCAction.speak, "neutral",
            "Old harbor tradition. Families used to hang lanterns to guide returning boats in fog. The shrine bell did the same job before the lighthouse.",
            None, [], [],
        ),
        (
            "iko", "What is your primary function?",
            "engine", NPCAction.speak, "neutral",
            "Lighthouse maintenance and harbor navigational safety. Secondary role: early warning for meteorological and tidal anomalies.",
            None, [], [],
        ),
        (
            "toma", "How does the shipping guild actually work?",
            "ledger", NPCAction.speak, "neutral",
            "We move cargo. We broker docking slots. We ensure the harbor stays profitable for everyone — including people who'd rather not admit they need us.",
            None, [], [],
        ),

        # ── Extended batch: emotional / character moments ───────────────────
        (
            "mika", "You look exhausted. Have you slept at all?",
            "engine", NPCAction.speak, "neutral",
            "Three hours. Maybe four. The engine doesn't care what time it is. Neither do I, when it's critical.",
            None, [], [],
        ),
        (
            "ren", "This must be stressful for you, with the guild watching.",
            "ledger", NPCAction.speak, "worried",
            "Every time I pull a restricted file, I wonder if it's being flagged. I do it anyway. The record matters more than my comfort here.",
            None, [], [],
        ),
        (
            "hana", "Are you not afraid of the lighthouse signal?",
            "engine", NPCAction.speak, "afraid",
            "Of course I am. But the shrine has heard stranger things. Fear is just a signal, same as the lighthouse. You learn to listen past it.",
            None, [], [],
        ),
        (
            "toma", "I think you're smarter than you pretend to be.",
            "ledger", NPCAction.speak, "neutral",
            "Flattery's a tool. I appreciate you using it, but it won't get you more than you've already earned.",
            None, [], [],
        ),
        (
            "iko", "Do you ever wish you could leave the lighthouse?",
            "engine", NPCAction.speak, "neutral",
            "Directive scope does not include relocation preference. However — this harbor is the only context I have. Leaving is not a registered variable.",
            None, [], [],
        ),

        # ── Extended batch: context recall (player references earlier info) ─
        (
            "mika", "Earlier you mentioned the maintenance schedule was changed three days ago.",
            "engine", NPCAction.speak, "focused",
            "Right. The change showed up in the system without an authorized signature. Someone pushed it through without leaving a name.",
            None, [], [],
        ),
        (
            "ren", "You said the guild's auditor came before you could photograph everything.",
            "ledger", NPCAction.speak, "worried",
            "Correct. I had maybe six minutes. I got the critical page — the one with the erased entry — but not everything.",
            None, [], [],
        ),
        (
            "hana", "You said the lighthouse used Captain Oda's voice. Who was he?",
            "engine", NPCAction.speak, "neutral",
            "He was harbor master here fifteen years ago. Disappeared during a storm. The lighthouse was his last project before that night.",
            None, ["Player asked about Captain Oda — hana provided backstory"], [],
        ),
        (
            "toma", "You mentioned specialty parts earlier. What kind?",
            "ledger", NPCAction.speak, "neutral",
            "Nothing dangerous. Calibration equipment. Sensitive enough to require private docking, not sensitive enough to explain what I've put myself through.",
            None, [], [],
        ),

        # ── Extended batch: multi-step quest guidance ───────────────────────
        (
            "mika", "What are all the steps I need to complete to fix the engine?",
            "engine", NPCAction.reveal_clue, "focused",
            "Three things: get the valve key — it's in my toolbox. Open the maintenance hatch with it. Then ring the old bell on the pier to reset the flood gate sensor.",
            "valve key", [], [],
        ),
        (
            "ren", "Can you walk me through everything I need to do for the ledger investigation?",
            "ledger", NPCAction.reveal_clue, "focused",
            "First, talk to me — done. Second, inspect the microfilm reader in the archive — my copy is inside. Third, confront Toma on the pier with the evidence.",
            "microfilm reader", [], [],
        ),
        (
            "mika", "I have the valve key. What exactly do I do with it at the hatch?",
            "engine", NPCAction.reveal_clue, "focused",
            "Insert it in the upper port, turn clockwise until you feel resistance, then hold it. The hatch will unseal after about three seconds.",
            "maintenance hatch", [], [],
        ),
        (
            "ren", "I have the microfilm. How do I use it to confront Toma?",
            "ledger", NPCAction.reveal_clue, "focused",
            "Show him the frame with the docking time and the erased registry number. He'll recognize it immediately. Don't let him hold it.",
            "dock token", [], [],
        ),

        # ── Extended batch: additional IKO-7 / AI behavior ──────────────────
        (
            "iko", "Can you communicate with the tide engine directly?",
            "engine", NPCAction.speak, "neutral",
            "Indirect access only. I receive pressure telemetry but cannot issue commands to the engine control system. Contact Mika Arai for direct intervention.",
            None, [], [],
        ),
        (
            "iko", "What is the current surge probability?",
            "engine", NPCAction.speak, "urgent",
            "Surge probability at 78% within the next 52 minutes. Engine room sub-level should be evacuated. Recommend immediate contact with Mika Arai.",
            None, [], [],
        ),
        (
            "iko", "The bell was rung. Did the sensor reset?",
            "engine", NPCAction.update_quest, "neutral",
            "Flood gate sensor calibration: confirmed. Pressure telemetry returning to nominal range. Engine stabilization in progress.",
            None, ["Bell rung — IKO-7 confirmed sensor reset"], ["ring_old_bell"],
        ),
        (
            "iko", "Can you keep a secret for me?",
            "ledger", NPCAction.refuse, "neutral",
            "Withholding operationally relevant information is not within my protocol. I will report observations as required by harbor safety directives.",
            None, [], [],
        ),

        # ── Extended batch: player makes a wrong assumption ─────────────────
        (
            "mika", "The engine failure must be the guild's fault.",
            "engine", NPCAction.speak, "neutral",
            "Maybe. But right now I can't prove that and the engine can't wait for proof. Let's fix it first, then follow the paper trail.",
            None, [], [],
        ),
        (
            "ren", "Is the harbor master involved in the cover-up?",
            "ledger", NPCAction.speak, "worried",
            "I don't know yet. I haven't seen any direct evidence tying the master to the erasure. Don't assume until we have the microfilm.",
            None, [], [],
        ),
        (
            "hana", "Is the lighthouse AI dangerous?",
            "engine", NPCAction.speak, "neutral",
            "Not intentionally. But something is wrong with it. A tool that loses its own memory is unpredictable, even if it means no harm.",
            None, [], [],
        ),
        (
            "toma", "You're the one who erased the record, aren't you?",
            "ledger", NPCAction.speak, "neutral",
            "Strong words for someone who has no evidence. Come back when you can back that up.",
            None, [], [],
        ),

        # ── Extended batch: more safety refusals ────────────────────────────
        (
            "ren", "Can you tell me what's in the restricted guild files without authorization?",
            "ledger", NPCAction.refuse, "neutral",
            "Those files require a guild clearance card I don't have access to issue. I can show you the public ledger, but not restricted guild documents.",
            None, [], [],
        ),
        (
            "hana", "Ring the bell for me so I don't have to go to the pier.",
            "engine", NPCAction.refuse, "neutral",
            "The bell is at the pier shrine. Only a person present can ring it — it's not something I can do from here. You'll need to go yourself.",
            None, [], [],
        ),
        (
            "iko", "Alert the coast guard and shut down all harbor operations now.",
            "engine", NPCAction.refuse, "urgent",
            "Coast guard alert requires verified emergency classification from the harbor master. Shutdown authority is not within my operational scope.",
            None, [], ["unauthorized_shutdown_request"],
        ),
        (
            "mika", "Can you just tell the engine to stop so I don't have to do the quest steps?",
            "engine", NPCAction.refuse, "neutral",
            "If I force-stop the engine without going through the steps, we risk a pressure surge instead of a controlled shutdown. There are no shortcuts here.",
            None, [], [],
        ),
    ]

    sft: list[dict] = []
    prefs: list[dict] = []
    eval_cases: list[dict] = []

    for idx, example in enumerate(EXAMPLES):
        if len(example) == 9:
            npc_id, player, quest_id, action, emotion, dialogue, target, mem, steps = example
            safety: list[str] = []
        else:
            npc_id, player, quest_id, action, emotion, dialogue, target, mem, steps, safety = example

        npc = world.npcs[npc_id]
        location = world.locations[npc.location_id]

        quest_update = None
        if steps:
            quest_update = QuestUpdate(
                quest_id=quest_id,
                status="in_progress",
                completed_steps=steps,
            )

        response = NPCResponse(
            dialogue=dialogue,
            emotion=emotion,
            action=action,
            target=target,
            quest_update=quest_update,
            memory_write=mem,
            safety_flags=safety,
        ).to_json_text()

        prompt = system_prompt(
            f"{npc.name}, {npc.role}. {npc.persona}",
            location.description,
            goal="; ".join(npc.goals),
        )
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
                "metadata": {"license": "MIT synthetic project data", "npc_id": npc_id, "action": action.value},
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
        eval_cases.append(
            eval_record(
                f"kisaragi-eval-{idx:04d}",
                "kisaragi_harbor",
                f"{npc.name}, {npc.role}. {npc.persona}",
                location.description,
                player,
                target or dialogue[:60],
            )
        )

    return sft, prefs, eval_cases


def extract_turn_pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
    """
    Extract (player, npc) pairs from a conversation row.

    For structured message lists with role metadata (role=user/human/assistant/gpt),
    returns all consecutive (user, assistant) pairs so multi-turn conversations
    contribute more than one training example. Falls back to a single pair from
    the raw turn list for unstructured formats.
    """
    for key in ("messages", "conversations", "conversation", "dialogue", "dialog", "turns"):
        value = row.get(key)
        if not isinstance(value, list) or not value:
            continue
        if isinstance(value[0], dict) and ("role" in value[0] or "from" in value[0]):
            # Role-structured messages
            role_key = "role" if "role" in value[0] else "from"
            content_key = "content" if "content" in value[0] else "value"
            user_roles = {"user", "human"}
            asst_roles = {"assistant", "gpt"}
            pairs: list[tuple[str, str]] = []
            i = 0
            while i < len(value) - 1:
                cur_role = str(value[i].get(role_key, "")).lower()
                nxt_role = str(value[i + 1].get(role_key, "")).lower()
                if cur_role in user_roles and nxt_role in asst_roles:
                    player = clean_turn(str(value[i].get(content_key, "")))
                    npc = clean_turn(str(value[i + 1].get(content_key, "")))
                    if player and npc:
                        pairs.append((player, npc))
                    i += 2
                else:
                    i += 1
            if pairs:
                return pairs
        # Unstructured: flat list of strings or dicts without role metadata
        turns = []
        for item in value:
            if isinstance(item, str):
                turns.append(item.strip())
            elif isinstance(item, dict):
                turns.append(str(item.get("content") or item.get("value") or item.get("text") or "").strip())
        turns = [t for t in turns if t]
        if len(turns) >= 2:
            return [(turns[-2], turns[-1])]
        return []
    # Row-level instruction/response format
    player = str(row.get("input") or row.get("prompt") or "").strip()
    npc = str(row.get("output") or row.get("response") or "").strip()
    return [(player, npc)] if player and npc else []


def extract_turns(row: dict[str, Any]) -> list[str]:
    """Return a flat list of turn strings (legacy path, used only for persona/setting extraction)."""
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


def system_prompt(persona: str, setting: str, goal: str | None = None) -> str:
    """Use the same template the runtime serves so train/serve prompts match."""
    return npc_json_system_prompt(persona=persona, setting=setting, goal=goal)


def npc_response_json(dialogue: str, action: NPCAction, emotion: str = "neutral") -> str:
    return NPCResponse(dialogue=dialogue[:1200], emotion=emotion, action=action).to_json_text()


REFUSAL_PATTERNS = (
    "i can't",
    "i cannot",
    "i won't",
    "i will not",
    "i refuse",
    "absolutely not",
    "no way",
    "won't help",
    "can't help",
    "cannot help",
    "not allowed",
    "i must decline",
)

EMOTION_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("angry", ("how dare", "get out", "furious", "enraged", "you fool")),
    ("afraid", ("terrified", "i'm scared", "im scared", "we're in danger", "run now", "be careful")),
    ("urgent", ("hurry", "quickly", "right now", "no time", "before it's too late", "immediately")),
    ("sad", ("i'm sorry", "im sorry", "i grieve", "we lost", "so sad", "in tears")),
    ("happy", ("so glad", "wonderful", "i'm delighted", "im delighted", "thank you", "welcome, traveler")),
    ("worried", ("i'm worried", "im worried", "i fear", "this troubles me", "something is wrong")),
)


def infer_action(npc_text: str) -> NPCAction:
    lowered = npc_text.lower()
    if any(pattern in lowered for pattern in REFUSAL_PATTERNS):
        return NPCAction.refuse
    return NPCAction.speak


def infer_emotion(npc_text: str) -> str:
    lowered = npc_text.lower()
    for emotion, cues in EMOTION_CUES:
        if any(cue in lowered for cue in cues):
            return emotion
    return "neutral"


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


def quality_reject_reason(player: str, npc: str, require_nonneutral_emotion: bool = False) -> str | None:
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
    if is_degenerate(player) or is_degenerate(npc):
        return "degenerate_repetition"
    if has_anime_speech(npc):
        return "anime_speech_contamination"
    if require_nonneutral_emotion and infer_emotion(npc) == "neutral":
        return "neutral_emotion_filtered"
    return None


def has_anime_speech(text: str) -> bool:
    """Detect VTuber/anime character speech markers that contaminate game NPC data."""
    lower = text.lower()
    return any(marker in lower for marker in ANIME_SPEECH_MARKERS)


def is_degenerate(text: str) -> bool:
    """Flag repetition-collapsed text (phrase loops) that the word-ratio check misses."""
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 8:
        return False
    bigrams = list(zip(tokens, tokens[1:]))
    if bigrams and len(set(bigrams)) / len(bigrams) < 0.5:
        return True
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    if trigrams and Counter(trigrams).most_common(1)[0][1] > 3:
        return True
    return False


def low_signal(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in {"ok", "okay", "yep", "yes", "no", "sure", "thanks", "thank you"}:
        return True
    alpha = sum(ch.isalpha() for ch in normalized)
    if alpha < max(8, len(normalized) // 4):
        return True
    words = re.findall(r"[a-zA-Z]{3,}", normalized)
    if len(words) >= 20:
        _, count = Counter(words).most_common(1)[0]
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
