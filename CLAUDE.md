# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research baseline for **Text-to-SQL** combining two innovations:
1. **Hybrid RAG Schema Linking**: Retrieves relevant schema elements using BM25 + ChromaDB semantic search + NPMI statistical scoring, fused via Reciprocal Rank Fusion (RRF).
2. **2-Stage SLM Fine-Tuning**: Fine-tunes Qwen3-4B with SFT (Stage 1) then GRPO reinforcement learning (Stage 2).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip install torchvision ijson
```

## Commands

**Run tests:**
```bash
pytest tests/
```

**Lint:**
```bash
ruff check src/ training/ scripts/
```

**Run single test:**
```bash
pytest tests/test_npmi_scorer.py -v
```

**Data preparation:**
```bash
# Merge OmniSQL datasets into SFT-ready JSONL
python scripts/prepare_sft_data.py \
    --data_dir datasets/data \
    --output_dir data/sft \
    --max_synsql 100000 \
    --thinking_ratio 0.75

# Build NPMI co-occurrence matrix
python scripts/build_npmi_matrix.py \
    --data_paths datasets/data/train_spider.json \
    --data_format omnisql \
    --output_path data/npmi_matrix.json
```

**Training:**
```bash
# Stage 1: SFT
torchrun --nproc_per_node=4 scripts/train_sft.py \
    --base_model ./models/Qwen3-4B \
    --data_source omnisql \
    --omnisql_data_paths data/sft/train.jsonl data/sft/dev.jsonl \
    --batch_size 2 --gradient_accumulation_steps 4 --num_epochs 3 \
    --load_in_4bit --output_dir ./checkpoints/sft

# Stage 2: GRPO RL
torchrun --nproc_per_node=4 scripts/train_rl.py \
    --sft_model_path ./checkpoints/sft/final \
    --train_data_path data/sft/train.jsonl
```

**End-to-end RAG inference + evaluation:**
```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

## Architecture

### RAG Inference Flow (`scripts/run_pipeline.py`)

```
Dataset → SpiderV1Adapter.load() → SchemaChunker.chunk_many() → SchemaIndexer (ChromaDB)

Per question:
  QueryAugmentor.augment()
    → HybridRetriever.retrieve()  [BM25 + semantic + NPMI → RRF]
    → BidirectionalLinker.expand()  [1-hop FK graph BFS]
    → SchemaFilter.filter_and_format()  [top-15 chunks → prompt string]
    → SQLInference.generate()  [Qwen3 outputs <think>...</think> + ```sql...```]
    → Evaluation (execution accuracy, exact match, schema recall/precision)
```

### Schema Chunk Types
Four types defined in `src/data/schema_chunker.py`: `table`, `column`, `fk`, `value`. Each chunk is embedded independently and stored in ChromaDB.

### Training Pipeline

**Stage 1 SFT** (`training/sft_trainer.py`): Uses Unsloth's `FastLanguageModel` with LoRA (rank=16, alpha=32) on Qwen3-4B. Complex SQL queries (JOINs, CTEs, subqueries, window functions, UNION, CASE WHEN, GROUP BY, HAVING) are always assigned thinking mode; others use a configurable ratio (default 75% thinking).

**Stage 2 GRPO** (`training/rl_trainer.py`): Multi-signal reward functions called independently:
- `check_sql_execution()` — weight 5.0 (execution accuracy)
- `match_sql_format_exactly()` — weight 3.0 (perfect `<think>+```sql``` format)
- `match_sql_format_approximately()` — weight 1.5 (partial format credit)
- `check_schema_faithfulness()` — weight 2.0 (only valid tables used)

DPO is also supported as an ablation via `RLConfig.algorithm = "dpo"`.

### Key Configuration

All pipeline parameters are in `configs/default.yaml`:
- `indexing.embedding_model`: `paraphrase-multilingual-mpnet-base-v2`
- `retrieval`: BM25/semantic top-k=30, RRF constant k=60, final top-k=15
- `npmi`: disabled by default; enable with `enabled: true` + `matrix_path`
- `training.sft.base_model`: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (default)
- `training.rl.algorithm`: `grpo` (default) or `dpo`

Dataclass equivalents live in `training/config.py` and are used by training scripts directly.

## Data Models

Defined in `src/data/base_adapter.py`:
- `Column`, `Table`, `Database` — schema objects
- `Example` — a question/SQL pair with `db_id`

`SpiderV1Adapter` (in `src/data/spider_v1_adapter.py`) loads Spider 1.0 format (tables.json + train/dev JSON). OmniSQL format datasets (BIRD, SynSQL-2.5M, EHRSQL, etc.) are loaded in `training/data_formatter.py` via `OmniSQLFormatter`, using `ijson` streaming for files >500MB.

## Supported Datasets

| Dataset | Size | Usage |
|---|---|---|
| Spider 1.0 | 7K train / 1K dev | Primary train/eval |
| BIRD | 9.4K train / 1.5K dev | OmniSQL format |
| SynSQL-2.5M | up to 200K subsample | Streaming via ijson |
| EHRSQL, ScienceBenchmark, Spider-DK/Syn/Realistic | dev sets | Evaluation |
