# Game NPC LLM Post-training Pipeline

Reproducible pipeline for building a quest-oriented game NPC model from public datasets only:

- LIGHT / LIGHT-WILD / LIGHT-Quests from ParlAI/Facebook Research.
- `chimbiwide/NPC-Dialogue_v2` from Hugging Face.

The project covers data conversion, QLoRA SFT with LLaMA-Factory, GRPO alignment with slime, SGLang serving, a LIGHT quest NPC agent, and evaluation through a custom lm-evaluation-harness task.

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

Optional server and RL extras:

```bash
pip install -e ".[train,serve,rl,eval]"
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
- `data/rl/grpo_prompts.jsonl`
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

Merge the main SFT adapter for serving/slime:

```bash
bash scripts/export_sft.sh
```

## GRPO

```bash
python3 scripts/prepare_slime_data.py
export SLIME_DIR=/path/to/slime
export SLIME_MODEL_ARGS_SCRIPT=/path/to/slime/scripts/models/qwen3.5-27B.sh
export REF_LOAD=outputs/slime/qwen3_5_27b_npc_ref_torch_dist
bash scripts/run_grpo.sh configs/slime/qwen3_5_27b_npc_grpo.sh
```

GRPO uses `data/rl/grpo_prompts.jsonl`, not the SFT assistant-answer dataset. Each RL row contains a quest prompt plus reward metadata; slime samples candidate NPC replies with SGLang and scores them through `game_npc_llm.rewards.slime_rm.reward_func`. Before running RL, convert the merged SFT Hugging Face checkpoint to slime/Megatron `torch_dist` format and point `REF_LOAD` at that directory.

After slime training, convert the saved Megatron checkpoint back to Hugging Face format with slime's `tools/convert_torch_dist_to_hf.py`, then place it at `outputs/grpo/qwen3_5_27b_npc_grpo_hf` or update `configs/sglang/models.yml`.

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

The configs follow the project assumption of `Qwen/Qwen3.5-27B` for the first full run and `Qwen/Qwen3.5-9B` for smoke tests.
