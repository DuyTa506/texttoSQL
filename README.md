# Text-to-SQL Experimental Space: RAG Schema Linking + Agentic Generation

This repository contains the baseline implementation for our Text-to-SQL research, combining Hybrid RAG Schema Linking with Small Language Model (SLM) Fine-Tuning.

## Project Structure

```
text2sql-baseline/
├── configs/            # YAML configuration files
├── data/               # Datasets, SFT output, NPMI matrices
├── notebooks/          # Exploratory Jupyter notebooks
├── scripts/            # CLI utilities
│   ├── run_pipeline.py          # End-to-end RAG inference
│   ├── prepare_sft_data.py      # Merge OmniSQL datasets → SFT-ready JSONL
│   ├── build_npmi_matrix.py     # Build NPMI co-occurrence matrix
│   ├── curate_dataset_kd.py     # Knowledge Distillation reasoning traces
│   └── build_golden_schema.py   # Golden Schema with noise injection
├── src/                # Core implementation
│   ├── data/           # Dataset loading, schema chunking, ChromaDB indexing
│   ├── retrieval/      # Hybrid BM25 + Semantic + NPMI, Bidirectional Linker
│   ├── generation/     # RAG Prompter and LLM Generator
│   └── evaluation/     # Execution accuracy and exact match metrics
├── training/           # Unsloth Training Module (SFT + RL)
│   ├── config.py       # Centralized hyperparameters
│   ├── sft_trainer.py  # Stage 1: SFT with LoRA (Qwen3 Thinking Mode)
│   ├── rl_trainer.py   # Stage 2: GRPO with vLLM sampling
│   ├── data_formatter.py # OmniSQL formatter + Qwen3 thinking format
│   └── reward.py       # Multi-signal GRPO reward functions
└── tests/              # Unit tests (NPMI, OmniSQL formatter)
```

## Core Features

1. **Hybrid Schema Retrieval**: BM25 + Semantic + **NPMI** (co-occurrence statistics) with Reciprocal Rank Fusion (RRF).
2. **Qwen3 Integration**: Leverages Qwen3's built-in thinking mode (`<think>...</think>`) with **complexity-aware think/direct** assignment.
3. **OmniSQL Multi-Dataset Training**: Merges Spider (7K) + BIRD (9.4K) + SynSQL (2.5M) into unified SFT data with streaming JSON support.
4. **Data Prep Pipeline**:
   - `prepare_sft_data.py`: Merges OmniSQL datasets → training-ready JSONL with `<think>` tags.
   - `build_npmi_matrix.py`: Builds NPMI co-occurrence matrix from training data.
   - `build_golden_schema.py`: Extracts gold schema with noise injection.
   - `curate_dataset_kd.py`: Teacher LLM reasoning traces (Knowledge Distillation).
5. **2-Stage SLM Training (Unsloth)**:
   - **Stage 1 (SFT)**: 75% thinking data + 25% direct, complexity-aware assignment (complex JOINs/CTEs always use thinking).
   - **Stage 2 (RL)**: GRPO with vLLM sampling and multi-signal rewards.

## Quick Start

### Installation

Navigate to the project directory and install the required dependencies:

```bash
pip install -e .[dev]
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
```

### Data Preparation

1. Download the OmniSQL datasets and extract to `OmniSQL/datasets/data/`.
2. Prepare merged SFT data (merge Spider + BIRD + SynSQL):
```bash
python scripts/prepare_sft_data.py \
    --data_dir OmniSQL/datasets/data \
    --output_dir data/sft \
    --max_synsql 200000
```
3. (Optional) Build NPMI matrix for retrieval:
```bash
python scripts/build_npmi_matrix.py \
    --data_paths OmniSQL/datasets/data/train_spider.json \
    --data_format omnisql \
    --output_path data/npmi_matrix.json
```
4. Enable NPMI in `configs/default.yaml`: set `npmi.enable: true`

### Training

Run the SFT trainer:
```bash
# Using pre-prepared JSONL (recommended)
python -m training.sft_trainer --data_source omnisql \
    --omnisql_data_paths data/sft/train.jsonl data/sft/dev.jsonl
```

### End-to-End Pipeline

```bash
python scripts/run_pipeline.py --dataset spider --mode run
```

## Supported Datasets

| Dataset | Source | Usage |
|---|---|---|
| Spider 1.0 | OmniSQL | Train (7K) + Dev (1K) |
| BIRD | OmniSQL | Train (9.4K) + Dev (1.5K) |
| SynSQL-2.5M | OmniSQL | Train (subsample) |
| EHRSQL | OmniSQL | Dev (1K) |
| ScienceBenchmark | OmniSQL | Dev (299) |
| Spider-DK/Syn/Realistic | OmniSQL | Dev (2K+) |

## Tech Stack

- **Training**: Unsloth + LoRA + TRL
- **Embeddings**: `paraphrase-multilingual-mpnet-base-v2`
- **Vector DB**: ChromaDB
- **Retrieval**: BM25 + Semantic + NPMI (RRF fusion)
- **SLM**: Qwen3-4B
- **RL**: GRPO (primary) / DPO (ablation)
- **Data**: `ijson` for streaming large JSON files
