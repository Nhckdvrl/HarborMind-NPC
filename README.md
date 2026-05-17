# Game NPC LLM Post-training Pipeline

Reproducible pipeline for building a quest-oriented game NPC model from public datasets only:

- LIGHT / LIGHT-WILD / LIGHT-Quests from ParlAI/Facebook Research.
- `chimbiwide/NPC-Dialogue_v2` from Hugging Face.

The project covers data conversion, QLoRA SFT with LLaMA-Factory, GRPO-LoRA alignment with verl, SGLang serving, a LIGHT quest NPC agent, and evaluation through a custom lm-evaluation-harness task.

## Repository Layout

```text
configs/                 Training, GRPO, serving, and eval configs
data/                    Raw and processed data output directory
scripts/                 Reproducible CLI entrypoints
src/game_npc_llm/        Data, agent, reward, serving, and eval code
tests/                   Local schema and state-machine tests
```

## Local Setup

```bash
conda create -p .venv-qwen35 python=3.11
conda activate "$(pwd)/.venv-qwen35"
pip install -e ".[dev]"
```

Optional local extras:

```bash
pip install -e ".[train,serve,eval]"
```

For GRPO training, install the RL extra in the training environment:

```bash
pip install -e ".[rl]"
```

## Data Pipeline

Dry run without downloading large datasets:

```bash
python3 scripts/download_data.py --dry-run
python3 scripts/local_smoke_test.py
```

Build processed JSONL files:

```bash
python3 scripts/download_data.py --output-dir data --seed 42
python3 scripts/validate_jsonl.py data/processed/sft_train.jsonl --schema sft
python3 scripts/validate_jsonl.py data/processed/grpo_prompts.jsonl --schema grpo
python3 scripts/validate_jsonl.py data/processed/eval_cases.jsonl --schema eval
```

Outputs:

- `data/processed/sft_train.jsonl`
- `data/processed/sft_validation.jsonl`
- `data/processed/sft_test.jsonl`
- `data/llamafactory/npc_sft_train.json`
- `data/llamafactory/npc_sft_valid.json`
- `data/raw_manifest.json`
- `data/processed/grpo_prompts.jsonl`
- `data/processed/eval_cases.jsonl`
- `data/eval/eval_cases.jsonl`

## SFT

Smoke:

```bash
bash scripts/run_sft.sh
```

Main:

```bash
bash scripts/run_sft.sh configs/llamafactory/qwen3_5_27b_npc_sft_qlora.yaml
```

LLaMA-Factory reads the local SFT JSONL files through `data/dataset_info.json`.
The processed `messages` column is registered as a ShareGPT/OpenAI-style chat dataset.
The Qwen3.5 configs use LLaMA-Factory's `qwen3_5_nothink` template, matching the direct NPC reply style in the SFT data.

Merge the main SFT adapter for serving and GRPO initialization:

```bash
bash scripts/export_sft.sh
```

## GRPO

```bash
python3 scripts/prepare_verl_grpo_data.py
```

Smoke on 100 LIGHT-Quests prompts:

```bash
bash scripts/run_verl_grpo.sh smoke
```

Full 27B run after smoke passes:

```bash
bash scripts/run_verl_grpo.sh full
```

GRPO uses verl-compatible parquet files under `data/rl/verl/`, not the SFT assistant-answer dataset. The smoke file is `data/rl/verl/grpo_smoke_100.parquet`; the full file is `data/rl/verl/grpo_train.parquet`. Each row stores the LIGHT-Quests prompt plus persona, setting, goal, allowed entities, expected actions, and source metadata in `extra_info`; verl calls `game_npc_llm.rewards.verl_reward.compute_score` for the rule reward.

## SGLang Serving

```bash
bash scripts/serve_sglang.sh base
bash scripts/serve_sglang.sh sft
bash scripts/serve_sglang.sh grpo
```

## Agent Demo

```bash
python3 scripts/run_agent_demo.py --model-url http://localhost:8000/v1 --model Qwen3.5-NPC-GRPO
```

## Evaluation

Validate task import locally:

```bash
python3 scripts/validate_lm_eval_task.py
```

Run a full eval on a served model:

```bash
lm_eval --model local-completions \
  --tasks game_npc_bench \
  --include_path src/game_npc_llm/eval/tasks \
  --model_args model=Qwen3.5-NPC-GRPO,base_url=http://localhost:8000/v1/completions \
  --output_path reports/grpo_eval.json
```

## Model Notes

The GRPO configs use `Qwen/Qwen3.5-27B` for both the 100-prompt smoke and the full run.
