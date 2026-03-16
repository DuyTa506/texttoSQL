# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research baseline for **Text-to-SQL** combining four pipeline phases with a novel **Schema Knowledge Graph** retriever:

1. **Phase 1 — Pre-Retrieval**: Value scanning (`ValueScanner`), question decomposition (`QuestionDecomposer`), and multi-strategy query augmentation.
2. **Phase 2 — Schema Linking**: Two interchangeable retrievers, both feature-flagged:
   - **HybridRetriever** (default): BM25 + ChromaDB semantic + NPMI → RRF, with FK-graph expansion.
   - **GraphRetriever** (new): Personalized PageRank over a typed `SchemaGraph` with 3-layer edges (structural + semantic + statistical) and LLM-enriched node descriptions/synonyms.
3. **Phase 3 — Generation**: Qwen3-4B with mode variants (`standard` / `cot_plan` / `divide_conquer`) and multi-candidate sampling.
4. **Phase 4 — Post-Generation**: Execution retry loop (run → classify error → re-prompt → repeat) and candidate selection via execution consistency.

Training uses a 2-stage SLM pipeline: SFT (Stage 1) then multitask GRPO reinforcement learning (Stage 2) that jointly trains nl2sql + SQL correction.

**All new phases are feature-flagged OFF by default** in `configs/default.yaml` — the original baseline is fully preserved.

---

## Setup

### With uv (recommended)

```bash
# Install uv if not already present
curl -Lsf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# or: pip install uv

# Create venv + install all core + dev deps in one step
uv sync --extra dev

# Add training deps (torch, transformers, trl, peft, accelerate …)
uv sync --extra train

# Unsloth must be installed separately (CUDA-specific wheel, not on PyPI)
uv pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
uv pip install torchvision ijson

# Optional: teacher LLM calls (correction data mining + node enrichment)
uv sync --extra correction
```

### With pip (fallback)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # core + dev
pip install -r requirements-train.txt      # + training deps
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip install torchvision ijson

# Optional: teacher LLM calls (correction data mining + node enrichment)
pip install openai anthropic      # adds openai + anthropic SDKs
```

---

## Commands

### Tests
```bash
pytest tests/                              # all tests
pytest tests/test_npmi_scorer.py -v
pytest tests/test_sql_executor.py -v
pytest tests/test_retry_loop.py -v
pytest tests/test_value_scanner.py -v
pytest tests/test_correction_formatter.py -v
```

### Lint
```bash
ruff check src/ training/ scripts/ llms/ embeddings/
```

### Schema Knowledge Graph (offline build)
```bash
# Structural + lexical edges only (no LLM, fast)
python scripts/build_schema_graph.py \
    --data_path data/spider \
    --output    data/schema_graphs/spider.json

# Full pipeline: enrichment + all 3 edge layers
python scripts/build_schema_graph.py \
    --data_path  data/spider \
    --db_dir     data/spider/database \
    --output     data/schema_graphs/spider_full.json \
    --enrich \
    --enrich_model gpt-4o-mini \
    --api_key  $OPENAI_API_KEY \
    --statistical \
    --value_overlap

# Resume: add statistical edges to an already-enriched graph
python scripts/build_schema_graph.py \
    --load_existing data/schema_graphs/spider_enriched.json \
    --data_path     data/spider \
    --output        data/schema_graphs/spider_full.json \
    --statistical

# Single database test
python scripts/build_schema_graph.py \
    --data_path  data/spider \
    --output     data/schema_graphs/concert_singer.json \
    --db_filter  concert_singer
```

### Data preparation
```bash
# Merge OmniSQL datasets into SFT-ready JSONL
python scripts/prepare_sft_data.py \
    --data_dir datasets/data \
    --output_dir data/sft \
    --max_synsql 100000 \
    --thinking_ratio 0.75

# Build NPMI co-occurrence matrix (for HybridRetriever NPMI signal)
python scripts/build_npmi_matrix.py \
    --data_paths datasets/data/train_spider.json \
    --data_format omnisql \
    --output_path data/npmi_matrix.json

# Mine SQL correction data (requires SFT checkpoint + API key)
python scripts/mine_correction_data.py \
    --sft_model_path ./checkpoints/sft/final \
    --data_path ./data/spider \
    --db_dir ./data/spider/database \
    --output_path ./data/correction/train.jsonl \
    --teacher_model gpt-4o \
    --max_samples 5000 \
    --difficulties medium hard
```

### Training
```bash
# Stage 1: SFT
torchrun --nproc_per_node=4 scripts/train_sft.py \
    --base_model ./models/Qwen3-4B \
    --data_source omnisql \
    --omnisql_data_paths data/sft/train.jsonl data/sft/dev.jsonl \
    --batch_size 2 --gradient_accumulation_steps 4 --num_epochs 3 \
    --load_in_4bit --output_dir ./checkpoints/sft

# Stage 2: GRPO RL (nl2sql only)
torchrun --nproc_per_node=4 scripts/train_rl.py \
    --sft_model_path ./checkpoints/sft/final \
    --train_data_path data/sft/train.jsonl

# Stage 2: GRPO RL (multitask: nl2sql + correction)
torchrun --nproc_per_node=4 scripts/train_rl.py \
    --sft_model_path ./checkpoints/sft/final \
    --train_data_path data/sft/train.jsonl \
    --correction_data_path data/correction/train.jsonl \
    --correction_mix_ratio 0.2
```

### End-to-end pipeline
```bash
# Baseline (HybridRetriever, all new phases disabled)
python scripts/run_pipeline.py --config configs/default.yaml

# GraphRetriever only (PPR schema linking)
python scripts/run_pipeline.py --config configs/default.yaml \
    --override schema_graph.enabled=true \
               schema_graph.graph_path=data/schema_graphs/spider_full.json

# GraphRetriever + HybridRetriever hybrid
python scripts/run_pipeline.py --config configs/default.yaml \
    --override schema_graph.enabled=true \
               schema_graph.graph_path=data/schema_graphs/spider_full.json \
               schema_graph.hybrid=true

# All phases enabled (graph retriever + value scan + cot_plan + retry)
python scripts/run_pipeline.py --config configs/default.yaml \
    --override schema_graph.enabled=true \
               schema_graph.graph_path=data/schema_graphs/spider_full.json \
               pre_retrieval.value_scan.enabled=true \
               pre_retrieval.decomposition.enabled=true \
               generation.mode=cot_plan \
               generation.n_candidates=3 \
               post_generation.retry_loop.enabled=true \
               post_generation.candidates.n=3
```

---

## Architecture

### 4-Phase Inference Flow (`scripts/run_pipeline.py`)

```
Dataset → SpiderV1Adapter.load() → [SchemaChunker → SchemaIndexer (ChromaDB)]
                                  → [SchemaGraphBuilder.build_many() (offline)]

Per question:
  ── PHASE 1: Pre-Retrieval (feature-flagged) ──────────────────────────────
  ValueScanner.scan()             [DB cell-value → NL fuzzy match]
  QuestionDecomposer.decompose()  [rule-based sub-question splitting]
  QueryAugmentor.augment()        [strategy: keyword | value | decompose]

  ── PHASE 2: Schema Linking ───────────────────────────────────────────────
  [default]  HybridRetriever.retrieve() / .retrieve_multi()
               BM25 + semantic (ChromaDB) + optional NPMI → RRF
               BidirectionalLinker.expand()  [1-hop FK BFS]
  [flagged]  GraphRetriever.retrieve() / .retrieve_multi()
               embed question → cosine seed scoring on KGNode.embedding
               synonym token boost → top-M seed nodes
               Personalized PageRank (alpha=0.7, max_hops=2)
               optional hybrid merge with HybridRetriever via RRF

  SchemaFilter.filter_and_format(value_hints=...)
    [top-15 nodes/chunks → CREATE TABLE prompt string + value hint comments]

  ── PHASE 3: Generation (feature-flagged) ─────────────────────────────────
  SQLInference.generate(mode=, n_candidates=)
    mode: standard | cot_plan | divide_conquer
    n_candidates > 1 → list[dict] with sampling at candidate_temperature

  ── PHASE 4: Post-Generation (feature-flagged) ────────────────────────────
  [n_candidates > 1] CandidateSelector.select()
    [majority vote on result-set hashes; tie-break: shortest SQL]
  [retry enabled]    RetryLoop.run()
    [execute → classify ErrorType → correction prompt → re-generate, ×3]

  ── Evaluation ────────────────────────────────────────────────────────────
  execution_accuracy, exact_match, schema_recall, schema_precision
  + retry_count, correction_applied (when Phase 4 active)
```

---

### Schema Knowledge Graph (`src/schema_graph/`)

A typed heterogeneous graph built from database schemas. Three layers of edges with clear separation of concerns:

#### Node types (`NodeType`)
| Type | node_id format | Content |
|---|---|---|
| `DATABASE` | `{db_id}` | Database-level node |
| `TABLE` | `{db_id}.{table}` | Table node with description + synonyms |
| `COLUMN` | `{db_id}.{table}.{col}` | Column node with dtype, PK/FK flags, sample values, description, synonyms, embedding |

#### Edge types by layer (`EdgeType`)
| Layer | EdgeType | Weight | Meaning |
|---|---|---|---|
| 1 | `DB_HAS_TABLE` | 1.0 | Database → Table ownership |
| 1 | `TABLE_HAS_COLUMN` / `COLUMN_BELONGS_TO` | 1.0 | Table ↔ Column membership (bidirectional) |
| 1 | `PRIMARY_KEY_OF` | 1.0 | PK column → Table annotation |
| 1 | `FOREIGN_KEY` / `FOREIGN_KEY_REV` | 0.95 | Column-level FK constraint (bidirectional) |
| 1 | `TABLE_FK` / `TABLE_FK_REV` | 0.85 | **v3**: Table-level FK shortcut (1-hop instead of 4-hop Table→Col→Col→Table) |
| 1 | `INFERRED_FK` / `INFERRED_FK_REV` | 0.95 | **v3**: Column FK inferred from high VALUE_OVERLAP (Jaccard ≥ 0.5) |
| 2a | `LEXICAL_SIMILAR` | ≤0.70 | Jaccard on name tokens (with stopword penalty) |
| 2b | `EMBEDDING_SIMILAR` | ≤0.90 | Cosine on LLM-enriched description vectors |
| 2b | `SYNONYM_MATCH` | ≤0.85 | Shared synonym tokens |
| 3 | `CO_JOIN` | ≤0.50 | Tables co-occur in FROM/JOIN (**v3**: capped 0.80→0.50) |
| 3 | `CO_PREDICATE` | ≤0.75 | Columns co-occur in WHERE/HAVING |
| 3 | `CO_SELECT` | ≤0.70 | Columns co-occur in SELECT list |
| 3 | `VALUE_OVERLAP` | ≤0.85 | Jaccard on DISTINCT SQLite cell values |

All edges are **bidirectional** (A→B and B→A both stored). Edge weight order: FK / INFERRED_FK (0.95) > EMBEDDING_SIMILAR (0.90) > TABLE_FK (0.85) > SYNONYM_MATCH (0.85) > VALUE_OVERLAP (0.85) > CO_PREDICATE (0.75) > LEXICAL_SIMILAR (0.70) > CO_JOIN (0.50).

#### PPR Retrieval (`GraphRetriever.retrieve()`)
```
1. Embed question → cosine score against all COLUMN + TABLE nodes with embeddings
2. Synonym token boost: nodes whose synonyms share tokens with the question get +0.3
3. Value seed boost (v3, if ValueScanner enabled):
   col_score += match.score * 0.5; tbl_score += match.score * 0.3
4. Top-M nodes selected as PPR seeds with personalisation vector ∝ cosine score
5. nx.pagerank(subgraph, alpha=0.7, personalization=seeds, weight="weight")
   — runs on per-DB subgraph only (isolated from other databases)
6. FK bridge injection (v3): add missing FK-target tables when FK columns are in results
7. Adaptive score-gap pruning (v3): cut tail where prev/curr score ratio > gap_ratio
   gap_ratio = 4.5 (≥10 tables) | 3.0 (5-9 tables) | 2.0 (≤4 tables)
8. Adaptive max_nodes cap: min(max_nodes, max(6, db_table_count))
9. Return nodes with final score > threshold, sorted descending
```

#### LLM Node Enrichment (`NodeEnricher`)
- Batches 15 nodes per API call (OpenAI or Anthropic), JSON-structured response
- Each node gets: `description` (1–2 sentences of business meaning) + `synonyms` (4–8 NL phrases)
- Fallback: individual retry per node if batch JSON parsing fails
- After enrichment: `embed_nodes()` computes SentenceTransformer embeddings on `"{name}. {description}. Synonyms: {syns}"` text
- Embeddings enable `EMBEDDING_SIMILAR` and `SYNONYM_MATCH` edges (Layer 2b)

#### Offline build cost estimate (Spider — 206 DBs, ~5K columns)
| Step | Time | API cost |
|---|---|---|
| Layer 1 (structural) | <1s | free |
| Layer 2a (lexical) | <5s | free |
| LLM enrichment (gpt-4o-mini, batch=15) | ~10 min | ~$1–3 |
| Node embedding | ~2 min | free |
| Layer 2b (embedding + synonym) | ~30s | free |
| Layer 3 statistical (co-occ + value overlap) | ~5 min | free |

---

### Post-Generation Error Classification (`src/post/sql_executor.py`)

`ErrorType` enum maps SQLite exceptions to specific correction hints:

| ErrorType | Trigger | Correction hint |
|---|---|---|
| `SUCCESS` | Results match gold | — |
| `SYNTAX_ERROR` | Parse failure | Check parentheses, keywords, quotes |
| `NO_SUCH_TABLE` | `no such table: X` | Use only schema tables |
| `NO_SUCH_COLUMN` | `no such column: X` | Check column names |
| `WRONG_RESULT` | Executes, wrong rows | Re-examine JOIN/WHERE logic |
| `EMPTY_RESULT` | 0 rows returned | Check WHERE / JOIN conditions |
| `EXECUTION_ERROR` | Other runtime errors | Type mismatches, division by zero |

---

### Training Pipeline

**Stage 1 SFT** (`training/sft_trainer.py`): Unsloth `FastLanguageModel` with LoRA (rank=16, alpha=32) on Qwen3-4B. Complex SQL (JOINs, CTEs, subqueries, window functions, UNION, CASE WHEN, GROUP BY, HAVING) always gets thinking mode; others use configurable ratio (default 75%).

**Stage 2 GRPO** (`training/rl_trainer.py`): Multi-signal reward functions called independently. Supports **multitask training** when `correction_data_path` is set — mixes nl2sql and correction samples at `correction_mix_ratio` (default 20%) per batch, with task-routed reward selection:

| Task | Reward functions |
|---|---|
| `nl2sql` | `match_sql_format_exactly` (3.0), `match_sql_format_approximately` (1.5), `check_sql_execution` (5.0), `check_schema_faithfulness` (2.0) |
| `correction` | `match_sql_format_exactly` (3.0), `check_sql_execution` (5.0), `check_correction_improvement` (4.0/2.0/-1.0), `check_error_addressed` (1.5) |

DPO is also supported as an ablation via `RLConfig.algorithm = "dpo"`.

---

### Key Configuration (`configs/default.yaml`)

**Original baseline flags (unchanged defaults):**
- `indexing.embedding_model`: `paraphrase-multilingual-mpnet-base-v2`
- `retrieval`: BM25/semantic top-k=30, RRF constant k=60, final top-k=15
- `npmi.enable`: `false` — enable with `true` + `matrix_path`
- `augmentation.strategy`: `keyword`

**New feature flags (all off by default):**
- `schema_graph.enabled`: `false` — set `true` to use `GraphRetriever`
- `schema_graph.graph_path`: `""` — path to built `.json` graph file
- `schema_graph.hybrid`: `false` — merge GraphRetriever + HybridRetriever via RRF
- `schema_graph.top_m`: `5` — PPR seed nodes per query
- `schema_graph.alpha`: `0.7` — PPR damping factor
- `schema_graph.max_hops`: `2` — PPR iteration depth
- `schema_graph.score_gap_ratio`: `3.0` — **v3**: adaptive gap pruning threshold (auto-adjusted per DB size)
- `schema_graph.use_fk_bridge`: `true` — **v3**: inject missing FK-target tables after PPR
- `pre_retrieval.value_scan.enabled`: `false`
- `pre_retrieval.decomposition.enabled`: `false`
- `generation.mode`: `standard`
- `generation.n_candidates`: `1`
- `post_generation.retry_loop.enabled`: `false`
- `post_generation.candidates.n`: `1`
- `training.rl.correction_data_path`: `""` (empty = nl2sql only)

Dataclass equivalents live in `training/config.py`. `RLConfig` has `correction_data_path` and `correction_mix_ratio` fields.

---

## Directory Reference

```
src/
  data/             base_adapter, spider_v1_adapter, schema_chunker, schema_indexer
  retrieval/        query_augmentor*, hybrid_retriever*, schema_filter*,
                    value_scanner [NEW], question_decomposer [NEW],
                    npmi_scorer, bidirectional_linker
  generation/       inference*
  post/             sql_executor [NEW], retry_loop [NEW],
                    candidate_selector [NEW], __init__ [NEW]
  evaluation/       metrics*, benchmark
  schema_graph/     [NEW PACKAGE]
    graph_types.py          KGNode, KGEdge, NodeType, EdgeType
    graph_builder.py        SchemaGraph (PPR + save/load + stats), SchemaGraphBuilder
    node_enricher.py        LLM descriptions + synonyms + SentenceTransformer embeddings
    graph_retriever.py      Drop-in for HybridRetriever (PPR + optional hybrid merge)
    edge_builders/
      structural_edges.py   Layer 1: DDL → FK/PK/membership edges
      semantic_edges.py     Layer 2: Jaccard / cosine / synonym edges
      statistical_edges.py  Layer 3: SQL co-occurrence + SQLite value overlap

llms/                              [NEW PACKAGE]
  base.py                BaseLLM ABC (generate, stream, agenerate, astream)
  openai.py              OpenAILLM — OpenAI API + any compatible base_url endpoint
  huggingface.py         HuggingFaceLLM — local AutoModelForCausalLM, lazy load, ChatML
  factory.py             LLMFactory.from_config() / .openai() / .huggingface() / .register()

embeddings/                        [NEW PACKAGE]
  base.py                BaseEmbeddingModel ABC (embed, embed_one, get_embedding_dimension)
  huggingface.py         HuggingFaceEmbeddingModel — sentence-transformers, auto device

training/
  config*           (+ correction_data_path, correction_mix_ratio in RLConfig)
  sft_trainer, data_formatter
  rl_trainer*       (+ multitask data mixing, _get_reward_funcs router)
  reward*           (+ check_correction_improvement, check_error_addressed)
  correction_formatter [NEW]

scripts/
  run_pipeline*           (4-phase wiring, --override CLI, GraphRetriever support)
  train_sft, train_rl*    (train_rl* = correction CLI args fixed)
  prepare_sft_data, build_npmi_matrix
  mine_correction_data [NEW]
  build_schema_graph   [NEW]   (offline 3-layer graph construction CLI)
  build_golden_schema, curate_dataset_kd

tests/
  test_npmi_scorer, test_omnisql_formatter  (existing)
  test_sql_executor [NEW]
  test_retry_loop [NEW]
  test_value_scanner [NEW]
  test_correction_formatter [NEW]

* = modified from baseline
[NEW] = added in this work
```

---

## LLM Provider Abstraction (`llms/`)

All LLM calls (teacher enrichment, correction mining, pipeline inference) go through `BaseLLM` — swap providers without touching calling code.

### Providers

| Provider class | Config `provider` key | Best for |
|---|---|---|
| `OpenAILLM` | `"openai"` | OpenAI API, Groq, Together.ai, Fireworks, Azure, local vLLM/LM Studio |
| `HuggingFaceLLM` | `"huggingface"` / `"hf"` / `"local"` | Fine-tuned checkpoints (SFT/RL), GPU inference |

`OpenAILLM` covers every OpenAI-compatible endpoint via `base_url`:
```python
# Standard OpenAI
llm = LLMFactory.openai("gpt-4o-mini")

# Groq (OpenAI-compatible)
llm = LLMFactory.openai("llama3-8b-8192",
      api_key=os.environ["GROQ_API_KEY"],
      base_url="https://api.groq.com/openai/v1")

# Local vLLM server
llm = LLMFactory.openai("Qwen/Qwen3-4B",
      api_key="token", base_url="http://localhost:8000/v1")

# Local SFT checkpoint (HuggingFace, lazy load)
llm = LLMFactory.huggingface("./checkpoints/sft/final", load_in_4bit=True)

# From config dict (YAML-driven)
llm = LLMFactory.from_config({"provider": "openai", "model_name": "gpt-4o-mini"})
```

`HuggingFaceLLM` is **lazy-loaded** — the model weights are not loaded until the first `generate()` call, so importing the class has no GPU memory cost.

### Embedding Model Abstraction (`embeddings/`)

`HuggingFaceEmbeddingModel` wraps `sentence-transformers` (already a core dep — no extra install):

```python
from embeddings import HuggingFaceEmbeddingModel

model = HuggingFaceEmbeddingModel()                            # default: paraphrase-multilingual-mpnet-base-v2
model = HuggingFaceEmbeddingModel("BAAI/bge-m3")              # swap model
vecs  = model.embed(["What are the top 5 products?"])          # list[str] → list[list[float]]
vec   = model.embed_one("What are the top 5 products?")        # str → list[float]
dim   = model.get_embedding_dimension()                        # int (768 for default)
```

Default model (`paraphrase-multilingual-mpnet-base-v2`) matches `SchemaIndexer` — no re-indexing needed when using the KG enrichment pipeline.

---

**Schema objects** (`src/data/base_adapter.py`):
- `Column(name, dtype, primary_key, description)` — column metadata
- `Table(name, columns)` — table with column list
- `ForeignKey(from_table, from_column, to_table, to_column)` — FK constraint
- `Database(db_id, tables, foreign_keys, sample_values, db_path)` — full schema; `db_path` used by `ValueScanner`, `SQLExecutor`, and `build_statistical_edges`
- `Example(db_id, question, query, difficulty)` — question/SQL pair

**Graph objects** (`src/schema_graph/graph_types.py`):
- `KGNode(node_id, node_type, db_id, table_name, column_name, dtype, is_pk, is_fk, sample_values, description, synonyms, embedding)` — enriched schema node
- `KGEdge(src_id, dst_id, edge_type, weight, metadata)` — typed weighted edge
- `NodeType`: DATABASE, TABLE, COLUMN
- `EdgeType`: DB_HAS_TABLE, TABLE_HAS_COLUMN, COLUMN_BELONGS_TO, PRIMARY_KEY_OF, FOREIGN_KEY, FOREIGN_KEY_REV, TABLE_FK, TABLE_FK_REV, INFERRED_FK, INFERRED_FK_REV, LEXICAL_SIMILAR, EMBEDDING_SIMILAR, SYNONYM_MATCH, CO_JOIN, CO_PREDICATE, CO_SELECT, VALUE_OVERLAP

**Training objects**:
- `CorrectionSample` + `CorrectionDataset` (`training/correction_formatter.py`) — correction training data serialisation and GRPO format conversion

---

## Supported Datasets

| Dataset | Size | Usage |
|---|---|---|
| Spider 1.0 | 7K train / 1K dev | Primary train/eval |
| BIRD | 9.4K train / 1.5K dev | OmniSQL format |
| SynSQL-2.5M | up to 200K subsample | Streaming via ijson |
| EHRSQL, ScienceBenchmark, Spider-DK/Syn/Realistic | dev sets | Evaluation |


<!-- ClaudeVibeCodeKit -->
## ClaudeVibeCodeKit

### Planning
When planning complex tasks:
1. Read `.claude/docs/plan-execution-guide.md` for format guide
2. Use planning-agent for parallel execution optimization
3. Output plan according to `.claude/schemas/plan-schema.json`

### Available Commands
- `/research <topic>` - Deep web research
- `/meeting-notes <name>` - Live meeting notes
- `/changelog` - Generate changelog
- `/onboard` - Developer onboarding
- `/handoff` - Create handoff document for conversation transition
- `/continue` - Resume work from a handoff document
- `/watzup` - Check current project status
- `/social-media-post` - Social content workflow
<!-- /ClaudeVibeCodeKit -->
