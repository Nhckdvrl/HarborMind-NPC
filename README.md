# Game NPC LLM Post-training Pipeline

Reproducible pipeline for building a quest-oriented game NPC model from public datasets only:

- LIGHT / LIGHT-WILD / LIGHT-Quests from ParlAI/Facebook Research.
- `chimbiwide/NPC-Dialogue_v2` from Hugging Face.

The project covers data conversion, QLoRA SFT with Axolotl, GRPO alignment with slime, vLLM serving, a LIGHT quest NPC agent, and evaluation through a custom lm-evaluation-harness task.

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
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional server extras:

```bash
pip install -e ".[train,serve,eval]"
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
- `data/processed/grpo_prompts.jsonl`
- `data/processed/eval_cases.jsonl`

## SFT

Smoke:

```bash
accelerate launch -m axolotl.cli.train configs/axolotl/qwen3_8b_npc_qlora_smoke.yml
```

Main:

```bash
accelerate launch -m axolotl.cli.train configs/axolotl/qwen3_32b_npc_qlora.yml
```

## GRPO

```bash
PORT=8091 bash scripts/run_reward_server.sh
bash scripts/run_grpo.sh configs/slime/qwen3_32b_npc_grpo.sh
```

The reward stack combines deterministic verifiers and optional OpenAI-compatible LLM judge scoring.

## vLLM Serving

```bash
bash scripts/serve_vllm.sh base
bash scripts/serve_vllm.sh sft
bash scripts/serve_vllm.sh grpo
```

## Agent Demo

```bash
python3 scripts/run_agent_demo.py --model-url http://localhost:8000/v1 --model Qwen3-NPC-GRPO
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
  --model_args model=Qwen3-NPC-GRPO,base_url=http://localhost:8000/v1/completions \
  --output_path reports/grpo_eval.json
```

## Model Notes

The configs follow the project assumption of `Qwen/Qwen3-32B` for the first full run and `Qwen/Qwen3-8B` for smoke tests. If your environment has `Qwen/Qwen3.5-27B` and `Qwen/Qwen3.5-4B` available, override `base_model` in the YAML files and model names in `configs/vllm/models.yml`.
