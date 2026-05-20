# GameNPC-RL

Product-oriented NPC LLM pipeline for a small RPG demo: public roleplay datasets, lightweight
LoRA SFT + ORPO/DPO alignment, structured JSON actions, long-term memory, FastAPI serving, and
a browser demo that can also be called from Unity.

The project is shaped for an NLP/game AI internship portfolio and focuses on a practical 3B-8B
workflow that can run on modest GPUs.

## What It Builds

- A `Qwen/Qwen3-4B` NPC model trained with LLaMA-Factory LoRA SFT.
- Optional ORPO alignment on chosen/rejected roleplay responses.
- A JSON-first NPC interface with `dialogue`, `emotion`, `action`, `target`, `quest_update`,
  `memory_write`, and `safety_flags`.
- A FastAPI service with `/chat`, `/world`, `/state/{session_id}`, `/memory/search`, and `/reset`.
- A browser demo for the original `Kisaragi Harbor` RPG scene.

## Data Strategy

Public datasets are the training backbone:

- `chimbiwide/RolePlay-NPC-Quest` as the primary NPC/quest roleplay source.
- `chimbiwide/NPC-Dialogue_v2` as an auxiliary game NPC dialogue source.
- `allenai/soda` as a downsampled social/emotional dialogue source.
- `vicgalle/OpenHermesPreferences-roleplay` as optional internal preference data because its
  license is not ideal for public model release.

The original `Kisaragi Harbor` data is intentionally small. It is used for demo, evaluation, and
schema grounding, not for pretending we hand-authored a huge training set.

## Setup

```bash
conda create -p .venv-qwen3 python=3.11
conda activate "$(pwd)/.venv-qwen3"
pip install -e ".[dev,serve,train]"
```

Optional extras:

```bash
pip install -e ".[eval,memory,rl]"
```

## Build Data

Tiny local smoke set:

```bash
python scripts/build_product_data.py --dry-run
python scripts/validate_jsonl.py data/processed/sft_train.jsonl --schema sft
python scripts/validate_jsonl.py data/processed/preference_train.jsonl --schema preference
python scripts/validate_jsonl.py data/processed/eval_cases.jsonl --schema eval
```

Full public-data build:

```bash
python scripts/build_product_data.py \
  --roleplay-limit 20000 \
  --npc-dialogue-limit 3500 \
  --soda-limit 10000
```

Add optional roleplay preference data:

```bash
python scripts/build_product_data.py --include-optional-preferences
```

## Train

SFT:

```bash
bash scripts/run_sft.sh
```

ORPO alignment:

```bash
bash scripts/run_sft.sh configs/llamafactory/qwen3_4b_npc_orpo_lora.yaml
```

Export merged SFT model:

```bash
bash scripts/export_sft.sh
```

## Serve And Demo

Rule-policy demo, no model server required:

```bash
uvicorn game_npc_llm.product.server:app --reload --port 8080
```

Open `http://localhost:8080`.

Connect a local llama.cpp or vLLM OpenAI-compatible endpoint:

```bash
NPC_MODEL_BASE_URL=http://localhost:8000/v1 \
NPC_MODEL_NAME=Qwen3-4B-NPC \
uvicorn game_npc_llm.product.server:app --port 8080
```

Unity can call the same `/chat` endpoint with:

```json
{
  "session_id": "unity-demo",
  "npc_id": "mika",
  "player_input": "The tide engine pressure is climbing."
}
```

## Evaluation

The benchmark scores product-facing behavior:

- JSON validity
- action validity
- role adherence
- quest progression
- memory write rate
- system leakage rate
- latency / throughput

The old lm-eval task remains lightweight, but the project direction is DeepEval-style scenario
evaluation plus deterministic rule checks.

## Research/Portfolio Narrative

This project demonstrates the full loop that a game R&D NLP intern is expected to understand:

1. Curate public roleplay/game dialogue data with licensing and filtering.
2. Train a small open model with LoRA SFT.
3. Improve behavior with modern offline preference optimization such as ORPO/SimPO/DPO.
4. Validate output with Pydantic schemas so a game engine can safely consume actions.
5. Add long-term memory and stateful quest execution.
6. Ship a usable Web/Unity-facing prototype.
