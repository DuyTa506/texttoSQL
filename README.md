# Text-to-SQL Experimental Space: RAG Schema Linking + Agentic Generation

This repository contains the baseline implementation for our Text-to-SQL research, combining Hybrid RAG Schema Linking with Small Language Model (SLM) Fine-Tuning.

## Project Structure

```
text2sql-baseline/
├── configs/            # YAML configuration files
├── data/               # Datasets and SQLite databases
├── notebooks/          # Exploratory Jupyter notebooks
├── scripts/            # CLI utilities (data prep, training, eval)
│   ├── run_pipeline.py # End-to-end RAG inference script
│   ├── curate_dataset_kd.py     # Knowledge Distillation for reasoning traces
│   └── build_golden_schema.py   # Golden Schema generator with noise injection
├── src/                # Core implementation components
│   ├── data/           # Dataset loading and vector DB indexing
│   ├── retrieval/      # Hybrid BM25 + Semantic, Bidirectional Schema Linker
│   ├── generation/     # RAG Prompter and LLM Generator
│   └── evaluation/     # Execution accuracy and exact match metrics
└── training/           # Unsloth Training Module (SFT + RL)
    ├── config.py       # Centralized hyperparameters
    ├── sft_trainer.py  # Stage 1: SFT with LoRA (Qwen3 Thinking Mode)
    ├── rl_trainer.py   # Stage 2: GRPO with vLLM sampling
    ├── data_formatter.py # Data prep for Qwen3 thinking format
    └── reward.py       # Multi-signal GRPO reward functions
```

## Core Features

1. **Hybrid Schema Retrieval**: Combines BM25 and SPLADE/SentenceTransformers with Reciprocal Rank Fusion (RRF).
2. **Qwen3-4B Integration**: Leverages Qwen3's built-in thinking mode (`<think>...</think>`) for step-by-step reasoning.
3. **Data Prep Pipeline**:
   - `build_golden_schema.py`: Extracts gold schema elements and injects negative samples (noise) to teach column discrimination.
   - `curate_dataset_kd.py`: Uses large teacher models (via LiteLLM) to generate high-quality reasoning traces based on the golden schema.
4. **2-Stage SLM Training (Unsloth)**:
   - **Stage 1 (SFT)**: Imparts reasoning format and basic structure. Uses 75% thinking data and 25% direct answering data.
   - **Stage 2 (RL)**: Applies GRPO (with vLLM sampling) utilizing multi-signal rewards (execution, format, schema faithfulness).

## Quick Start

### Installation

Navigate to the project directory and install the required dependencies:

```bash
pip install -e .[dev]
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
```

### Data Preparation

1. Download the Spider 1.0 dataset and extract it to `data/spider/`.
2. Generate Golden Schema Contexts (with noise tables):
```bash
python scripts/build_golden_schema.py \
    --dataset_path data/spider/train_spider.json \
    --tables_path data/spider/tables.json \
    --output_path data/schema_contexts_golden.json
```
3. (Optional) Extract reasoning traces using a Teacher LLM (Knowledge Distillation):
```bash
OPENAI_API_KEY="..." python scripts/curate_dataset_kd.py \
    --dataset_path data/spider/train_spider.json \
    --tables_path data/spider/tables.json \
    --schema_contexts_path data/schema_contexts_golden.json \
    --output_path data/reasoning_cache.json \
    --model gpt-4o-mini
```

### Training

Run the specialized modules within the `training/` directory to fine-tune the Qwen3-4B models.

### End-to-End Pipeline

To execute the full text-to-SQL evaluation pipeline on a specific dataset:

```bash
python scripts/run_pipeline.py --dataset spider --mode run
```

## Supported Datasets

| Dataset | Adapter | Status |
|---|---|---|
| Spider 1.0 | `spider_v1_adapter.py` | Ready (Baseline) |
| Spider 2.0 | `spider_v2_adapter.py` | Planned |
| BIRD | `bird_adapter.py` | Planned |

## Tech Stack

- **Training**: Unsloth + LoRA + TRL
- **Embeddings**: `paraphrase-multilingual-mpnet-base-v2` + `SPLADE`
- **Vector DB**: ChromaDB + Qdrant
- **SLM**: Qwen3-4B
- **RL**: GRPO (primary) / DPO (ablation)
