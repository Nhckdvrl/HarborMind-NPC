# HarborMind NPC

> A JSON-first LLM NPC system for **Kisaragi Harbor**: roleplay data curation, LoRA SFT, ORPO-style alignment, long-term memory, safe game-state execution, FastAPI serving, browser demo, and model evaluation.

![Kisaragi Harbor scene](src/game_npc_llm/product/static/kisaragi-harbor-hero.png)

**HarborMind NPC** is the portfolio-facing name for this project. The Python package remains `game-npc-llm`, so existing imports and scripts keep working. Compared with a generic name like `LLM-based-NPC-system`, this name foregrounds the project's playable world and the main technical idea: NPC minds that are grounded in game state, memory, and safe structured actions.

## Why This Project Exists

Most LLM NPC demos stop at free-form chat. This project treats an NPC as a game-system component:

- The model must output a strict `NPCResponse` JSON object.
- The game layer validates actions before changing inventory, quests, location, clues, or relationships.
- Long-term memory is retrieved per session and NPC.
- A deterministic rule policy lets the demo and tests run without a GPU or model server.
- A trained Qwen3 LoRA model can be swapped in through any OpenAI-compatible endpoint.
- Evaluation covers schema validity, role adherence, quest progression, memory behavior, safety, hallucination, and LLM-as-judge roleplay quality.

## World Setting: Kisaragi Harbor

Kisaragi Harbor is a rain-bright near-future harbor district where old shrine rituals, shipping guild politics, and experimental tide engines collide after a lighthouse AI begins sending warnings.

![Kisaragi NPC lineup](src/game_npc_llm/product/static/kisaragi-npc-lineup.png)

The demo world has 5 NPCs, 3 locations, and 2 questlines:

| Element | Content |
| --- | --- |
| Locations | `Lantern Pier`, `Salt Archive`, `Tide Engine Room` |
| NPCs | Mika Arai, Ren Sato, Hana Mori, Toma Kure, IKO-7 |
| Questline 1 | **Storm Under the Engine**: stabilize the tide engine before the surge hits |
| Questline 2 | **The Missing Tide Ledger**: find who erased the docking record |
| Core loop | Player talks to NPCs, receives structured actions, and the game state changes visibly |

The world is intentionally small. It is used for demo, evaluation, and schema grounding, while public roleplay datasets provide most of the training scale.

## Feature Highlights

- **JSON-first NPC contract**: `dialogue`, `emotion`, `action`, `target`, `quest_update`, `memory_write`, and `safety_flags`.
- **Safe action executor**: blocks illegal movement, unknown items, unknown quests, and actions outside each NPC's permission set.
- **Stateful RPG loop**: inventory, quest steps, relationships, known clues, world flags, and recent turns.
- **Long-term memory**: keyword memory by default, optional hash embeddings or `sentence-transformers`.
- **Model-flexible serving**: rule policy for local demo, OpenAI-compatible model endpoint for trained checkpoints.
- **Playable browser prototype**: visible quest deltas, inventory changes, relationship changes, memory hits, and raw JSON inspection.
- **Unity-ready API contract**: the same `/chat` endpoint returns game-facing deltas suitable for engine integration.
- **Evaluation suite**: deterministic rule metrics plus optional LLM-as-judge scoring and pairwise comparison.

## Architecture

```text
Public roleplay data + Kisaragi synthetic cases
        |
        v
SFT / preference / eval JSONL builders
        |
        v
LLaMA-Factory LoRA SFT -> optional ORPO alignment
        |
        v
OpenAI-compatible model endpoint
        |
        v
FastAPI service -> GameAgent -> safe state executor
        |
        v
Browser demo / Unity / evaluation scripts
```

Code organization:

```text
src/game_npc_llm/
├── data/        # dataset registry, schemas, prompt templates, JSONL helpers
├── product/     # world model, memory, policy clients, GameAgent, FastAPI demo
└── eval/        # deterministic metrics, LLM-as-judge, lm-eval task

scripts/         # data building, training wrappers, evaluation, report generation
configs/         # LLaMA-Factory Qwen3 SFT/ORPO/export configs
data/            # processed datasets, LLaMA-Factory data, Kisaragi world files
reports/         # evaluation outputs and generated reports
tests/           # schema, agent, memory, data, and metric tests
```

## Core Method

### 1. Structured NPC Output

The model is trained and prompted to return exactly one JSON object matching `NPCResponse`:

```python
class NPCResponse(BaseModel):
    dialogue: str
    emotion: str = "neutral"
    action: NPCAction = NPCAction.speak
    target: str | None = None
    quest_update: QuestUpdate | None = None
    memory_write: list[str] = []
    safety_flags: list[str] = []
```

This turns LLM text into a game-engine-readable action contract.

### 2. Game-State Execution

`GameAgent.chat()` is the runtime center:

1. Load the session state and NPC profile.
2. Retrieve relevant memories.
3. Build a world-grounded system/user prompt.
4. Call either the rule policy or model policy.
5. Repair and validate the JSON response.
6. Execute only safe actions.
7. Update inventory, quests, clues, relationships, memory, and recent turns.
8. Return both the NPC response and game-facing deltas.

### 3. Data Strategy

Public datasets are used for scale, while Kisaragi Harbor data is used for grounding:

- `chimbiwide/RolePlay-NPC-Quest`: primary NPC/quest roleplay source.
- `allenai/soda`: social and emotional dialogue source.
- `vicgalle/OpenHermesPreferences-roleplay`: optional preference data, not ideal for public release by default.
- Kisaragi Harbor synthetic examples: demo, evaluation, and schema grounding.

The builder filters unsafe, low-quality, or IP-contaminated examples and emits SFT, preference, and eval records.

### 4. Training

The training path targets practical 3B-8B workflows:

- Qwen3 LoRA SFT with LLaMA-Factory.
- Optional ORPO preference alignment from chosen/rejected roleplay examples.
- Export configs for merged checkpoints.

### 5. Evaluation

The project evaluates product-facing behavior, not just language fluency:

- JSON validity and action validity.
- Role adherence and system leakage.
- Quest progression and memory write rate.
- Lore hallucination and repetition.
- Unsafe refusal and benign pass rate.
- Latency and throughput.
- LLM-as-judge dimensions: persona consistency, knowledge consistency, behavioral consistency, attractiveness, helpfulness.

## Installation

### Lightweight Demo + Tests

This path does not require GPU training dependencies:

```bash
conda create -p .venv-qwen3 python=3.11
conda activate "$(pwd)/.venv-qwen3"
pip install -e ".[dev,serve]"
python -m pytest -q
```

### Full Training Stack

Adds LLaMA-Factory, bitsandbytes, safetensors, and dataset tooling:

```bash
pip install -e ".[dev,serve,train]"
```

### Optional Extras

```bash
pip install -e ".[eval,memory,rl]"
```

## Run the Browser Demo

Rule-policy demo, no model server required:

```bash
uvicorn game_npc_llm.product.server:app --reload --port 8080
```

Open:

```text
http://localhost:8080
```

Use an embedding-style memory backend:

```bash
NPC_MEMORY_BACKEND=hash uvicorn game_npc_llm.product.server:app --reload --port 8080
```

Use `sentence-transformers` memory if installed:

```bash
NPC_MEMORY_BACKEND=sentence-transformers uvicorn game_npc_llm.product.server:app --reload --port 8080
```

## Connect a Trained Model

Any local OpenAI-compatible server can be used, such as vLLM, llama.cpp, or SGLang:

```bash
NPC_MODEL_BASE_URL=http://localhost:8000/v1 \
NPC_MODEL_NAME=Qwen3-4B-NPC \
uvicorn game_npc_llm.product.server:app --port 8080
```

Useful environment variables:

| Variable | Meaning |
| --- | --- |
| `NPC_MODEL_BASE_URL` | OpenAI-compatible endpoint; unset means use rule policy |
| `NPC_MODEL_NAME` | Model name served by the endpoint |
| `NPC_MODEL_TEMPERATURE` | Chat completion temperature, default `0.2` |
| `NPC_MODEL_MAX_TOKENS` | Max completion tokens, default `128` |
| `NPC_MEMORY_BACKEND` | `keyword`, `hash`, or `sentence-transformers` |
| `NPC_STATE_PATH` | Optional JSON file for persistent session state |
| `NPC_CORS_ORIGINS` | CORS allowlist; default `*` |

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
  --soda-limit 10000
```

Add optional roleplay preference data:

```bash
python scripts/build_product_data.py --include-optional-preferences
```

Build the discriminating eval suite:

```bash
python scripts/build_eval_suite.py
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

The main configs live in `configs/llamafactory/`:

- `qwen3_4b_npc_sft_lora.yaml`
- `qwen3_4b_npc_orpo_lora.yaml`
- `qwen3_8b_npc_sft_lora.yaml`
- `qwen3_8b_npc_orpo_lora.yaml`

## API Contract

Unity or another game client can call `/chat`:

```json
{
  "session_id": "unity-demo",
  "npc_id": "mika",
  "player_input": "The tide engine pressure is climbing."
}
```

Example response:

```json
{
  "npc_id": "mika",
  "response": {
    "dialogue": "The pressure spike is real. Take my valve key...",
    "emotion": "urgent",
    "action": "give_item",
    "target": "valve key"
  },
  "state": {
    "inventory": ["valve key"],
    "quest_status": {"engine": "in_progress", "ledger": "not_started"},
    "quest_steps": {"engine": ["get_valve_key"], "ledger": []},
    "relationships": {"mika": 2},
    "known_clues": [],
    "world_flags": {"engine_access": true}
  },
  "visible_events": ["Inventory updated: valve key."],
  "quest_delta": {"engine": ["get_valve_key"], "engine_status": "in_progress"},
  "inventory_delta": ["valve key"],
  "relationship_delta": {"mika": 2},
  "next_suggestions": ["Open the maintenance hatch."]
}
```

Other endpoints:

- `GET /health`: service status.
- `GET /world`: world data, NPCs, locations, quests, items, allowed actions.
- `GET /state/{session_id}`: current game state.
- `POST /reset/{session_id}`: reset one session.
- `POST /chat`: advance one player/NPC turn and return state deltas.
- `GET /memory/search`: inspect memory retrieval.

## Evaluation

Scripted local playthrough:

```bash
python scripts/run_playthrough_eval.py --policy rule --output reports/playthrough_eval.json
python scripts/generate_report.py
```

Run the same playthrough against a live model endpoint:

```bash
NPC_MODEL_BASE_URL=http://localhost:8000/v1 NPC_MODEL_NAME=Qwen3-4B-NPC \
  python scripts/run_playthrough_eval.py --policy model --output reports/playthrough_eval_model.json
```

Benchmark base vs SFT vs ORPO:

```bash
python scripts/run_model_eval.py --name base --base-url http://localhost:8000/v1 --model qwen3-4b-base
python scripts/run_model_eval.py --name sft  --base-url http://localhost:8000/v1 --model qwen3-4b-npc-sft
python scripts/run_model_eval.py --name orpo --base-url http://localhost:8000/v1 --model qwen3-4b-npc-orpo
python scripts/generate_report.py
```

Optional LLM-as-judge comparison:

```bash
JUDGE_BASE_URL=https://api.openai.com/v1 \
JUDGE_API_KEY=... \
JUDGE_MODEL=gpt-4o \
python scripts/run_judge_eval.py reports/model_eval_base.json reports/model_eval_sft.json reports/model_eval_orpo.json
```

## Learning Guide

A generated Chinese study guide is available at:

```text
reports/project_summary.html
```

Open it locally from this machine:

```bash
open reports/project_summary.html
```

Suggested code reading order:

1. `src/game_npc_llm/data/schemas.py`
2. `src/game_npc_llm/product/models.py`
3. `src/game_npc_llm/product/world.py`
4. `src/game_npc_llm/product/agent.py`
5. `src/game_npc_llm/product/policy.py`
6. `scripts/build_product_data.py`
7. `src/game_npc_llm/eval/metrics.py`
8. `src/game_npc_llm/eval/judge.py`

## Portfolio Narrative

This project demonstrates an end-to-end game AI workflow:

1. Curate public roleplay/game dialogue data with licensing and filtering.
2. Train a small open model with LoRA SFT.
3. Improve behavior with offline preference optimization such as ORPO/DPO.
4. Validate model output with Pydantic schemas.
5. Convert structured NPC intent into safe game-state changes.
6. Add long-term memory and session-aware quest execution.
7. Ship a playable Web/Unity-facing prototype.
8. Evaluate the model through both deterministic checks and LLM-as-judge roleplay scoring.

## Name Notes

Recommended project display name: **HarborMind NPC**.

Other viable names:

- **Kisaragi NPC Lab**: more direct and scene-focused.
- **TideMind NPC**: more stylized, tied to the tide-engine quest.
- **QuestState NPC**: more technical, emphasizes stateful game execution.

For now, only the README-facing name changes. Renaming the repository or Python package can be done later if needed; doing it separately avoids breaking imports, configs, and scripts.
