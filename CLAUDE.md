# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Text-to-SQL research combining Hybrid RAG Schema Linking with a 2-stage SLM fine-tuning pipeline (SFT → GRPO). Base model: **Qwen3.5-2B** with built-in `<think>...</think>` reasoning mode.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
pip install --upgrade --force-reinstall "transformers==5.5.0"
pip install torchvision ijson
```

**Critical:** `unsloth` and `unsloth_zoo` must be from the same source. A version mismatch causes `KeyError: 'sanitize_logprob'`. Fix: add `sanitize_logprob` to `unsloth_zoo/rl_replacements.py` (see README.md).

## Commands

```bash
# Lint
ruff check .

# Tests
pytest tests/
pytest tests/test_npmi.py  # single test file

# Prepare SFT data (merge Spider + BIRD + SynSQL)
python scripts/prepare_sft_data.py --data_dir datasets/data --output_dir data/sft --max_synsql 100000

# Stage 1: SFT training (4-GPU DDP)
torchrun --nproc_per_node=4 scripts/train_sft.py \
    --base_model ./models/Qwen3.5-2B \
    --data_source omnisql \
    --omnisql_data_paths data/sft/train.jsonl data/sft/dev.jsonl \
    --batch_size 2 --gradient_accumulation_steps 4 --num_epochs 3 \
    --output_dir ./checkpoints/sft

# Prepare RL data (5-stage curation pipeline)
python scripts/prepare_rl_data.py \
    --data_dir datasets/data \
    --sft_model_path ./checkpoints/sft/final \
    --num_candidates 5 --teacher_model gpt-4o-mini \
    --output_dir data/rl

# Stage 2: GRPO training
torchrun --nproc_per_node=4 scripts/train_rl.py \
    --sft_model_path ./checkpoints/sft/final \
    --train_data_path data/rl/grpo_train.jsonl \
    --output_dir ./checkpoints/rl

# RAG schema retrieval pipeline
python scripts/run_pipeline.py --config configs/default.yaml

# Knowledge distillation (teacher reasoning traces)
python scripts/curate_dataset_kd.py \
    --dataset_path data/spider/train_spider.json \
    --tables_path data/spider/tables.json \
    --output_path data/reasoning_cache.json \
    --model gpt-4o-mini

# Build NPMI matrix (optional, improves retrieval)
python scripts/build_npmi_matrix.py --data_paths datasets/data/train_spider.json \
    --data_format omnisql --output_path data/npmi_matrix.json
```

Effective SFT batch size = `batch_size × grad_accum × num_GPUs`. Example: 2 × 4 × 4 = **32**.

## Architecture

### Two-Stage Training Pipeline

```
OmniSQL datasets (Spider 7K + BIRD 9.4K + SynSQL 2.5M)
        │
        ▼  scripts/prepare_sft_data.py
data/sft/train.jsonl  (instruction + response with <think> blocks)
        │
        ▼  scripts/train_sft.py → training/sft_trainer.py
checkpoints/sft/final  (LoRA adapter on Qwen3.5-2B)
        │
        ▼  scripts/prepare_rl_data.py  (5-stage pipeline)
data/rl/grpo_train.jsonl  (prompt + answer + db_path + task_type)
        │
        ▼  scripts/train_rl.py → training/rl_trainer.py
checkpoints/rl/final  (GRPO-refined model)
```

### RL Data Curation (5 Stages in `scripts/prepare_rl_data.py`)

1. **Holdout Identification** — SynSQL examples beyond the SFT cutoff (primary source), filtered to medium→hard complexity
2. **Candidate Generation** — SFT model generates 3–5 SQL candidates per question
3. **Execution Split** — Real SQLite execution via `src/evaluation/metrics.execution_accuracy()`; EX=True → GRPO shard, all-fail → Stage 4
4. **Teacher Error Correction** — Failed cases sent to teacher LLM (litellm) with `<Broken SQL> + <Error Msg> + <Gold SQL>` → structured `<think>` diagnosis + corrected SQL
5. **Assembly** — Two task types in `grpo_train.jsonl`: `task_type="text2sql"` and `task_type="error_correction"`

### GRPO Reward Functions (`training/rl_trainer.py`)

Six signals, all auto-forwarded from dataset columns via `trl.GRPOTrainer`:

| Function | Max Score | Signal |
|---|---|---|
| `match_sql_format_exactly` | 3.0 | Full `</think>...```sql...``` ` format |
| `match_sql_format_approximately` | 1.5 | Partial format + thinking-collapse penalty (`-0.5` if `<think>` < 5 words) |
| `check_sql_execution_real` | 5.0 | Real SQLite execution vs gold; falls back to string match if no `db_path` |
| `check_column_set_matching` | 2.0 | Dense structural reward: table Jaccard + identifier Jaccard (SQL-ASTRA CSMR) |
| `check_schema_faithfulness` | 2.0 | Only references tables from schema |
| `check_error_correction_success` | 6.8 | Bonus only for `task_type=error_correction`; 0.0 for standard tasks |

### Hybrid RAG Schema Retrieval (`src/retrieval/`)

Three signals fused via Reciprocal Rank Fusion (RRF, k=60):
- **BM25** (sparse, `rank-bm25`)
- **Semantic** (dense, ChromaDB + `paraphrase-multilingual-mpnet-base-v2`)
- **NPMI** (statistical co-occurrence, optional — enable in `configs/default.yaml`)

Post-retrieval: `BidirectionalLinker` expands results along FK chains; `SchemaFilter` trims to top-K.

### Key Data Formats

**SFT training** (`data/sft/train.jsonl`):
```json
{"instruction": "Given the SQLite database schema...\nSchema:\n...\nQuestion: ...",
 "response": "<think>\nreasoning\n</think>\n\n```sql\nSELECT ...\n```"}
```

**GRPO training** (`data/rl/grpo_train.jsonl`):
```json
{"prompt": "...", "answer": "SELECT ...", "schema_tables": "t1,t2",
 "task_type": "text2sql", "db_path": "datasets/data/spider/database/db/db.sqlite"}
```

Error-correction examples add `"task_type": "error_correction"` and include the broken SQL in the prompt.

### Complexity-Aware Thinking Assignment (`scripts/prepare_sft_data.py`)

`is_complex_sql()` (regex on JOIN/CTE/HAVING/GROUP BY/etc.) determines thinking mode:
- Complex queries → **always** `<think>` mode
- Simple queries → `<think>` mode with `--thinking_ratio` probability (default 0.75)
- Dev data → 100% `<think>` mode

SynSQL has an explicit `sql_complexity` field (`Simple/Moderate/Complex/Highly Complex`) used instead of the regex when available.

### Centralized Hyperparameters (`training/config.py`)

Three dataclasses: `LoRAConfig` (r=32, targets all linear layers), `SFTConfig` (Qwen3.5-2B, 4096 seq len, lr=2e-4), `RLConfig` (GRPO, grpo_num_generations=4, lr=5e-6, kl_coeff=0.05). CLI args in the train scripts override these defaults.

## Important Notes

- **OmniSQL format**: Raw datasets have `{input_seq, output_seq}` format. `parse_input_seq()` / `parse_output_seq()` extract schema+question and reasoning+SQL respectively. The OmniSQL files do NOT contain `db_id` — use index-aligned original source files to get `db_id` and `db_path`.
- **Database paths** by source:
  - Spider: `datasets/data/spider/database/{db_id}/{db_id}.sqlite`
  - BIRD: `datasets/data/bird/train/train_databases/{db_id}/{db_id}.sqlite`
  - SynSQL: `datasets/data/SynSQL-2.5M/databases/{db_id}/{db_id}.sqlite`
- **Large files**: `train_synsql.json` is ~22GB; always use `load_json_streaming()` (ijson) for it.
- **`reward.py` vs `rl_trainer.py`**: `training/reward.py` has a `RewardFunction` class for evaluation/DPO. The GRPO reward functions in `rl_trainer.py` are the ones actually wired into training.
- **EvolKit** (`EvolKit/`): Instruction-evolution framework (WizardLM-style). Not part of the main pipeline; available for data augmentation experiments. Uses its own OpenAI-SDK-based generators (not litellm).
- **Knowledge Distillation** (`scripts/curate_dataset_kd.py`): Uses `litellm.acompletion` for multi-provider teacher models. Output is `{db_id__{md5(question)[:8]}: reasoning_text}` — keyed to match `DataFormatter`.

## Shadow File Technique

When making large or multi-location edits to a file, use the shadow file approach:
1. Create `filename.ext.shadow` — write the complete final state directly (do not copy)
2. If too large, append sections sequentially
3. Verify syntax: `python3 -c "import ast; ast.parse(open('file.shadow').read())"`
4. Replace: `rm filename.ext && mv filename.ext.shadow filename.ext`

This preserves Git history and avoids partial-edit errors.
