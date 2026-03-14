# Text-to-SQL Research Baseline: Schema Knowledge Graph + 4-Phase RAG Pipeline

A research baseline for Text-to-SQL combining a **Schema Knowledge Graph retriever**, a **4-phase inference pipeline**, and a **2-stage SLM training pipeline** (SFT → multitask GRPO).

All new components are **feature-flagged OFF by default** — the original baseline behaviour is fully preserved and any combination can be enabled independently via `--override` flags.

---

## Project Structure

```
text2sql-baseline/
├── configs/
│   └── default.yaml                  # All pipeline flags (phases 1–4, schema graph, training)
├── data/                             # Datasets, SFT output, NPMI matrices, correction data
│   └── schema_graphs/               # Built graph JSON files (offline, pre-built)
├── scripts/
│   ├── run_pipeline.py              # 4-phase end-to-end pipeline + --override CLI
│   ├── build_schema_graph.py        # [NEW] Offline 3-layer schema graph construction
│   ├── train_sft.py                 # Stage 1: SFT (DDP-compatible)
│   ├── train_rl.py                  # Stage 2: GRPO / DPO (DDP-compatible)
│   ├── prepare_sft_data.py          # Merge OmniSQL datasets → SFT-ready JSONL
│   ├── build_npmi_matrix.py         # Build NPMI co-occurrence matrix
│   ├── mine_correction_data.py      # [NEW] Mine SQL correction data via teacher LLM
│   ├── curate_dataset_kd.py         # Knowledge Distillation reasoning traces
│   └── build_golden_schema.py       # Golden schema with noise injection
├── src/
│   ├── data/                        # Dataset loading, schema chunking, ChromaDB indexing
│   ├── retrieval/
│   │   ├── query_augmentor.py       # Multi-strategy: keyword | value | decompose
│   │   ├── hybrid_retriever.py      # BM25 + Semantic + NPMI → RRF (+retrieve_multi)
│   │   ├── schema_filter.py         # Top-K filter + value hint injection
│   │   ├── value_scanner.py         # [NEW] DB cell-value → NL fuzzy matching
│   │   ├── question_decomposer.py   # [NEW] Rule-based sub-question splitting
│   │   ├── npmi_scorer.py           # NPMI co-occurrence statistics
│   │   └── bidirectional_linker.py  # FK graph BFS expansion
│   ├── schema_graph/                # [NEW PACKAGE] Schema Knowledge Graph
│   │   ├── graph_types.py           # KGNode, KGEdge, NodeType, EdgeType
│   │   ├── graph_builder.py         # SchemaGraph (PPR + save/load) + SchemaGraphBuilder
│   │   ├── node_enricher.py         # LLM node descriptions + synonyms + embeddings
│   │   ├── graph_retriever.py       # Drop-in for HybridRetriever (PPR + hybrid mode)
│   │   └── edge_builders/
│   │       ├── structural_edges.py  # Layer 1: DDL → FK / PK / membership
│   │       ├── semantic_edges.py    # Layer 2: Jaccard / cosine / synonym
│   │       └── statistical_edges.py # Layer 3: SQL co-occurrence + value overlap
│   ├── generation/
│   │   └── inference.py             # SQLInference (+mode, +n_candidates)
│   ├── post/                        # [NEW PACKAGE]
│   │   ├── sql_executor.py          # Execute SQL + classify ErrorType
│   │   ├── retry_loop.py            # Error → correction prompt → re-generate loop
│   │   └── candidate_selector.py   # Majority vote on result-set hashes
│   └── evaluation/
│       └── metrics.py               # EX, EM, schema recall/precision (+retry metrics)
├── llms/                             # [NEW] LLM provider abstraction
│   ├── base.py                      # BaseLLM ABC
│   ├── openai.py                    # OpenAI + every compatible endpoint (Groq, Together, vLLM…)
│   ├── huggingface.py               # Local HF / Unsloth checkpoints (AutoModelForCausalLM)
│   └── factory.py                   # LLMFactory.from_config() / .openai() / .huggingface()
├── embeddings/                       # [NEW] Embedding model abstraction
│   ├── base.py                      # BaseEmbeddingModel ABC
│   └── huggingface.py               # sentence-transformers provider (no extra deps)
├── training/
│   ├── config.py                    # Dataclass hyperparams (+correction fields in RLConfig)
│   ├── sft_trainer.py               # Stage 1: SFT with Unsloth + LoRA
│   ├── rl_trainer.py                # Stage 2: GRPO (+multitask mixing, +reward router)
│   ├── reward.py                    # Reward functions (+correction rewards)
│   ├── correction_formatter.py      # [NEW] CorrectionSample + CorrectionDataset
│   └── data_formatter.py            # OmniSQL formatter + Qwen3 thinking format
└── tests/
    ├── test_npmi_scorer.py
    ├── test_omnisql_formatter.py
    ├── test_sql_executor.py          # [NEW]
    ├── test_retry_loop.py            # [NEW]
    ├── test_value_scanner.py         # [NEW]
    └── test_correction_formatter.py  # [NEW]
```

---

## Pipeline Architecture

```
Dataset → SpiderV1Adapter → [SchemaChunker → SchemaIndexer (ChromaDB)]   ← HybridRetriever path
                           → [SchemaGraphBuilder.build_many()  (offline)] ← GraphRetriever path

Per question:
┌─ PHASE 1: Pre-Retrieval ─────────────────────────────────── (feature-flagged) ─┐
│  ValueScanner.scan()              DB cell-value → NL fuzzy match               │
│  QuestionDecomposer.decompose()   rule-based, deterministic, zero-latency      │
│  QueryAugmentor.augment()         keyword | value | decompose strategy         │
└─────────────────────────────────────────────────────────────────────────────────┘
┌─ PHASE 2: Schema Linking ────────────────────────────────── (always active) ───┐
│  [default] HybridRetriever    BM25 + semantic + NPMI → RRF                     │
│                               BidirectionalLinker 1-hop FK expansion           │
│  [flagged] GraphRetriever     Personalized PageRank over SchemaGraph           │
│                               LLM-enriched node embeddings + synonym boost     │
│                               optional hybrid merge with HybridRetriever       │
│  SchemaFilter.filter_and_format(value_hints=)  top-15 → prompt string          │
└─────────────────────────────────────────────────────────────────────────────────┘
┌─ PHASE 3: Generation ────────────────────────────────────── (feature-flagged) ─┐
│  SQLInference.generate(mode=, n_candidates=)                                   │
│    standard        — default single-pass                                       │
│    cot_plan        — query-planner system instruction                          │
│    divide_conquer  — CTE/subquery decomposition instruction                    │
│    n_candidates>1  — diverse sampling → list[dict]                             │
└─────────────────────────────────────────────────────────────────────────────────┘
┌─ PHASE 4: Post-Generation ───────────────────────────────── (feature-flagged) ─┐
│  CandidateSelector.select()   majority vote on result-set hashes               │
│  RetryLoop.run()              execute → ErrorType → correction prompt → retry  │
└─────────────────────────────────────────────────────────────────────────────────┘
  Evaluation: EX, EM, schema_recall, schema_precision, retry_count
```

---

## Schema Knowledge Graph

The `SchemaGraph` is a typed heterogeneous graph over database schema elements with three independently-built layers of edges:

### Node Types

| Type | node_id | Carries |
|---|---|---|
| `DATABASE` | `{db_id}` | Top-level database node |
| `TABLE` | `{db_id}.{table}` | Description, synonyms |
| `COLUMN` | `{db_id}.{table}.{col}` | dtype, PK/FK flags, sample values, description, synonyms, embedding vector |

### Edge Layers

| Layer | EdgeType | Weight | Source |
|---|---|---|---|
| **1 — Structural** | `TABLE_CONTAINS` | 1.00 | DDL (table ↔ column membership) |
| 1 | `FK_REFERENCE` | 0.95 | DDL foreign key |
| 1 | `PK_COLUMN` | 0.90 | DDL primary key |
| 1 | `SAME_TABLE` | 0.50 | Sibling columns in same table |
| **2 — Semantic** | `LEXICAL_SIMILAR` | 0.70 | Jaccard on name tokens (no LLM needed) |
| 2 | `EMBEDDING_SIMILAR` | 0.90 | Cosine on enriched description embeddings |
| 2 | `SYNONYM_MATCH` | 0.85 | Shared synonym tokens from LLM enrichment |
| **3 — Statistical** | `CO_JOIN` | 0.80 | Tables co-occur in FROM/JOIN (training SQL) |
| 3 | `CO_PREDICATE` | 0.75 | Columns co-occur in WHERE/HAVING |
| 3 | `CO_SELECT` | 0.70 | Columns co-occur in SELECT list |
| 3 | `VALUE_OVERLAP` | 0.85 | Jaccard on DISTINCT SQLite cell values |

All edges are bidirectional. Layers 1 and 2a (lexical) require no API calls and build in seconds.

### Retrieval: Personalized PageRank (PPR)

```
1. Embed question → cosine score vs. all COLUMN + TABLE node embeddings
2. Synonym token boost for nodes whose synonyms match question tokens
3. Top-M nodes become PPR seed set with personalisation proportional to cosine score
4. Power iteration (max_hops=2, alpha=0.7) propagates relevance along edge weights
5. Return nodes above score threshold, ranked, capped at max_nodes=20
```

### LLM Node Enrichment (one-time offline)

`NodeEnricher` batches 15 nodes per API call to generate:
- **description**: 1–2 sentences of business meaning (e.g. `hire_date` → *"The date the employee joined the company. Used in tenure calculations."*)
- **synonyms**: 4–8 NL phrases a user might say (e.g. `["hired", "start date", "joining date", "employment date"]`)

After enrichment, `embed_nodes()` computes `"{name}. {description}. Synonyms: {syns}"` embeddings via SentenceTransformer, enabling `EMBEDDING_SIMILAR` and `SYNONYM_MATCH` edges.

**Cost for Spider (206 DBs, ~5K columns):** ~$1–3 using `gpt-4o-mini`, ~10 min total.

### Ablation matrix supported out of the box

| Config | What it tests |
|---|---|
| `HybridRetriever` only (default) | BM25 + semantic baseline |
| `GraphRetriever` (Layer 1+2a only) | Graph structure, no enrichment |
| `GraphRetriever` (all 3 layers, enriched) | Full KG approach |
| `GraphRetriever` + `HybridRetriever` (hybrid) | Fusion of both |

---

## Error-Type → Correction Hint Mapping (Phase 4)

| ErrorType | SQLite trigger | Hint injected into re-prompt |
|---|---|---|
| `SYNTAX_ERROR` | Parse failure | Check parentheses, keywords, quotes |
| `NO_SUCH_TABLE` | `no such table: X` | Use only tables from the schema |
| `NO_SUCH_COLUMN` | `no such column: X` | Check column names in the schema |
| `WRONG_RESULT` | Executes, wrong rows | Re-examine JOIN conditions and filters |
| `EMPTY_RESULT` | 0 rows returned | Check WHERE / JOIN eliminating rows |
| `EXECUTION_ERROR` | Other runtime errors | Type mismatches, division by zero |

---

## Quick Start

### Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip install torchvision ijson

# Optional: LLM API calls (node enrichment + correction data mining)
pip install -e ".[correction]"   # adds openai + anthropic SDKs
```

> **⚠️ Unsloth version warning:** Installing `unsloth` from GitHub while `unsloth_zoo` comes from PyPI can cause a `KeyError: 'sanitize_logprob'`. Fix by adding to `unsloth_zoo/rl_replacements.py`:
> ```python
> def sanitize_logprob(logprob):
>     if logprob is None: return logprob
>     return torch.nan_to_num(logprob, nan=0.0, posinf=0.0, neginf=0.0)
> RL_REPLACEMENTS["sanitize_logprob"] = sanitize_logprob
> ```

### Download Base Model

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B', local_dir='./models/Qwen3-4B')
"
```

---

## Data Preparation

```bash
# 1. Merge OmniSQL datasets → SFT-ready JSONL
python scripts/prepare_sft_data.py \
    --data_dir datasets/data \
    --output_dir data/sft \
    --max_synsql 100000 \
    --thinking_ratio 0.75

# 2. (Optional) Build NPMI matrix for HybridRetriever statistical signal
python scripts/build_npmi_matrix.py \
    --data_paths datasets/data/train_spider.json \
    --data_format omnisql \
    --output_path data/npmi_matrix.json
# Then enable: npmi.enable: true in configs/default.yaml

# 3. (Optional) Build Schema Knowledge Graph — fast, no LLM
python scripts/build_schema_graph.py \
    --data_path data/spider \
    --output    data/schema_graphs/spider.json

# 3b. (Optional) Build Schema Knowledge Graph — full (LLM enrichment + all edges)
python scripts/build_schema_graph.py \
    --data_path    data/spider \
    --db_dir       data/spider/database \
    --output       data/schema_graphs/spider_full.json \
    --enrich \
    --enrich_model gpt-4o-mini \
    --statistical \
    --value_overlap

# 4. (Optional) Mine correction data for multitask GRPO
#    Requires: SFT checkpoint + OpenAI or Anthropic API key
python scripts/mine_correction_data.py \
    --sft_model_path ./checkpoints/sft/final \
    --data_path ./data/spider \
    --db_dir ./data/spider/database \
    --output_path ./data/correction/train.jsonl \
    --teacher_model gpt-4o \
    --max_samples 5000 \
    --difficulties medium hard
```

---

## Training

### Stage 1 — SFT

```bash
torchrun --nproc_per_node=4 scripts/train_sft.py \
    --base_model ./models/Qwen3-4B \
    --data_source omnisql \
    --omnisql_data_paths data/sft/train.jsonl data/sft/dev.jsonl \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3 \
    --load_in_4bit \
    --output_dir ./checkpoints/sft
```

> Effective batch size = `batch_size × grad_accum × num_GPUs` (e.g. 2 × 4 × 4 = **32**).

### Stage 2 — GRPO Reinforcement Learning

```bash
# nl2sql only (baseline)
torchrun --nproc_per_node=4 scripts/train_rl.py \
    --sft_model_path ./checkpoints/sft/final \
    --train_data_path data/sft/train.jsonl

# Multitask: nl2sql + SQL correction (80/20 mix)
torchrun --nproc_per_node=4 scripts/train_rl.py \
    --sft_model_path ./checkpoints/sft/final \
    --train_data_path data/sft/train.jsonl \
    --correction_data_path data/correction/train.jsonl \
    --correction_mix_ratio 0.2
```

**Multitask reward routing:**

| Task type | Reward functions |
|---|---|
| `nl2sql` | `match_sql_format_exactly` · `match_sql_format_approximately` · `check_sql_execution` · `check_schema_faithfulness` |
| `correction` | `match_sql_format_exactly` · `check_sql_execution` · `check_correction_improvement` · `check_error_addressed` |

---

## Inference & Evaluation

### Baseline (all new phases disabled)
```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

### A/B Ablation Examples

```bash
# Schema graph retriever only (PPR, no enrichment)
python scripts/run_pipeline.py --config configs/default.yaml \
    --override schema_graph.enabled=true \
               schema_graph.graph_path=data/schema_graphs/spider.json

# Schema graph retriever (fully enriched, all 3 layers)
python scripts/run_pipeline.py --config configs/default.yaml \
    --override schema_graph.enabled=true \
               schema_graph.graph_path=data/schema_graphs/spider_full.json

# Hybrid: graph PPR + BM25/semantic merged via RRF
python scripts/run_pipeline.py --config configs/default.yaml \
    --override schema_graph.enabled=true \
               schema_graph.graph_path=data/schema_graphs/spider_full.json \
               schema_graph.hybrid=true

# Phase 1: value scanning only
python scripts/run_pipeline.py --config configs/default.yaml \
    --override pre_retrieval.value_scan.enabled=true

# Phase 4: retry loop only
python scripts/run_pipeline.py --config configs/default.yaml \
    --override post_generation.retry_loop.enabled=true

# All phases combined
python scripts/run_pipeline.py --config configs/default.yaml \
    --override schema_graph.enabled=true \
               schema_graph.graph_path=data/schema_graphs/spider_full.json \
               schema_graph.hybrid=true \
               pre_retrieval.value_scan.enabled=true \
               pre_retrieval.decomposition.enabled=true \
               generation.mode=cot_plan \
               generation.n_candidates=3 \
               post_generation.retry_loop.enabled=true \
               post_generation.candidates.n=3
```

The `--override KEY=VALUE` flag accepts any dot-path into `configs/default.yaml`. Values are auto-cast to `bool` / `int` / `float` / `str`.

---

## Testing

```bash
# All tests
pytest tests/ -v

# Schema graph (new)
pytest tests/test_sql_executor.py tests/test_retry_loop.py \
       tests/test_value_scanner.py tests/test_correction_formatter.py -v

# Regression: existing tests
pytest tests/test_npmi_scorer.py tests/test_omnisql_formatter.py -v

# Lint (includes new llms/ and embeddings/ packages)
ruff check src/ training/ scripts/ llms/ embeddings/
```

---

## Supported Datasets

| Dataset | Source | Size | Usage |
|---|---|---|---|
| Spider 1.0 | OmniSQL | 7K train / 1K dev | Primary train/eval |
| BIRD | OmniSQL | 9.4K train / 1.5K dev | SFT training |
| SynSQL-2.5M | OmniSQL | up to 200K subsample | Streaming via ijson |
| EHRSQL | OmniSQL | 1K dev | Evaluation |
| ScienceBenchmark | OmniSQL | 299 dev | Evaluation |
| Spider-DK/Syn/Realistic | OmniSQL | 2K+ dev | Evaluation |

---

## Tech Stack

| Component | Technology |
|---|---|
| SLM | Qwen3-4B (built-in `<think>` mode) |
| Training | Unsloth + QLoRA + TRL (`torchrun` DDP) |
| RL Algorithm | GRPO (primary) / DPO (ablation) |
| LLM API (teacher/enrichment) | OpenAI / Groq / Together / local vLLM via `OpenAILLM` |
| Local LLM inference | HuggingFace `AutoModelForCausalLM` via `HuggingFaceLLM` |
| Embeddings | `paraphrase-multilingual-mpnet-base-v2` via `HuggingFaceEmbeddingModel` |
| Vector DB | ChromaDB (embedded, no Docker) |
| Schema Graph | Custom typed heterogeneous graph + PPR (NetworkX-free, pure Python) |
| Graph edges | Structural (DDL) + Semantic (Jaccard/cosine/synonym) + Statistical (SQL co-occ/value overlap) |
| Node enrichment | OpenAI / Anthropic API (offline, one-time) |
| Retrieval | BM25 (`rank_bm25`) + Semantic + NPMI → RRF  **or**  Graph PPR |
| SQL Execution | `sqlite3` (stdlib) |
| Data Streaming | `ijson` (for files >500MB) |
| SQL Parsing | `sqlparse` |

---

## Hardware Requirements

| Setup | VRAM per GPU | Mode |
|---|---|---|
| 1× GPU ≥16GB | 16GB+ | QLoRA (`--load_in_4bit`) |
| 4× GPU ≥16GB | 16GB each | DDP + QLoRA (recommended) |
| 1× GPU ≥24GB | 24GB+ | Full LoRA (FP16) |
