# Handoff Document

**Created:** 2026-03-17
**Topic:** `src/` structural cleanup — folder layout + import hygiene
**Status:** Completed — all changes pushed, 226/226 tests passing
**Repo:** `https://github.com/DuyTa506/texttoSQL.git`
**Branch:** `main` (clean working tree, fully pushed)

---

## Context Summary

Two structural refactors were applied this session on top of the Phase 1–3 work from `handoff-20260317-phase3.md`.
No algorithm or behavior changes — all 226 tests pass after every commit.
Two commits pushed: `7e05fdf` and `066c96a`.

---

## Completed Work — This Session

### Folder restructure (`7e05fdf`)

**Problem:** `src/retrieval/` was a catch-all mixing Phase 1 pre-retrieval components, Phase 2 retrievers, and internal utilities.

**Solution:**

| Old location | New location | Reason |
|---|---|---|
| `src/retrieval/value_scanner.py` | `src/pre_retrieval/value_scanner.py` | Phase 1 component |
| `src/retrieval/question_decomposer.py` | `src/pre_retrieval/question_decomposer.py` | Phase 1 component |
| `src/retrieval/query_augmentor.py` | `src/pre_retrieval/query_augmentor.py` | Phase 1 component |
| `src/retrieval/npmi_scorer.py` | `src/retrieval/utils/npmi_scorer.py` | HybridRetriever internal signal |
| `src/retrieval/bidirectional_linker.py` | `src/retrieval/utils/bidirectional_linker.py` | HybridRetriever FK expansion |
| `src/retrieval/schema_filter.py` | `src/retrieval/utils/schema_filter.py` | Prompt formatter |

`src/retrieval/` now contains **only** retrievers: `BaseRetriever`, `HybridRetriever`, `GraphRetriever`.

New package entry points:
- `src/pre_retrieval/__init__.py` — exports `ValueMatch`, `ValueScanner`, `QuestionDecomposer`, `QueryAugmentor`
- `src/retrieval/utils/__init__.py` — exports `NPMIScorer`, `BidirectionalLinker`, `SchemaFilter`

Import sites updated: `run_pipeline.py`, `analyze_retrieval.py`, `mine_correction_data.py`, `build_npmi_matrix.py`, `benchmark.py`, `test_value_scanner.py`, `test_npmi_scorer.py`, `hybrid_retriever.py`.

Relative imports inside `retrieval/utils/` bumped from `..schema.X` → `...schema.X` (one extra depth level).

---

### Schema `__init__` + flat imports (`066c96a`)

**Problem:** `src/schema/__init__.py` was empty, forcing all callers to do deep relative imports like `from ...schema.models import Database` or `from ..schema.schema_chunker import SchemaChunk`, where the number of dots varies by file depth.

**Solution:** Populated `src/schema/__init__.py` to re-export everything:

```python
from .models import Column, Database, Example, ForeignKey, Table
from .schema_chunker import SchemaChunk, SchemaChunker
from .schema_indexer import SchemaIndexer
```

All deep relative imports across `src/` replaced with a single flat form:
```python
from src.schema import Database, SchemaChunk   # works at any depth, no dot counting
```

Files updated: `src/data_parser/base.py`, `bird_parser.py`, `spider_parser.py`, `src/evaluation/benchmark.py`, `src/pre_retrieval/query_augmentor.py`, `src/pre_retrieval/value_scanner.py`, `src/retrieval/hybrid_retriever.py`, `src/retrieval/utils/bidirectional_linker.py`, `src/retrieval/utils/npmi_scorer.py`, `src/retrieval/utils/schema_filter.py`.

---

## Current `src/` Structure

```
src/
  pre_retrieval/              ← Phase 1 (NEW package)
    __init__.py               ← exports ValueScanner, QuestionDecomposer, QueryAugmentor
    value_scanner.py
    question_decomposer.py
    query_augmentor.py

  retrieval/                  ← Phase 2: retrievers only
    __init__.py               ← exports BaseRetriever, HybridRetriever, GraphRetriever
    base_retriever.py
    hybrid_retriever.py
    graph_retriever.py
    utils/                    ← NEW subfolder
      __init__.py             ← exports NPMIScorer, BidirectionalLinker, SchemaFilter
      npmi_scorer.py
      bidirectional_linker.py
      schema_filter.py

  schema/
    __init__.py               ← NOW POPULATED: exports all public classes
    models.py
    schema_chunker.py
    schema_indexer.py

  shared/
    sqlite_utils.py           ← safe_execute() helper
  llms/
    chat_format.py
    factory.py                ← + LLMConfig dataclass
  evaluation/
    __init__.py
    benchmark.py              ← + BenchmarkReporter class
  schema_graph/               ← graph construction only (no retriever here)
  generation/
  post/
  data_parser/
```

---

## Pending Work — Phase 3 Deferred (wait for API stability)

From `handoff-20260317-phase3.md` — still open:

- [ ] `@dataclass RetrievalResult(id, chunk, score, source)` — replace raw `list[dict]` across all retrievers + callers. **High risk**, touches every retriever, schema_filter, run_pipeline, benchmark, tests.
- [ ] Decompose `HybridRetriever` into composed `BM25Retriever` + `SemanticRetriever`. **High risk**, same reason.

---

## Pending Work — GraphRetriever v3 Algorithm (highest priority)

From `handoff-20260316-1118.md` — still open, **this is the next thing to work on**:

| Priority | Task | Expected gain |
|---|---|---|
| **HIGH** | Table-level PPR seeds for FK columns | +3–5 pp recall |
| **HIGH** | Per-DB adaptive `score_gap_ratio` (≥10 tables → 4.5, 5–9 → 3.0, ≤4 → 2.0) | −50+ noisy examples |
| **MEDIUM** | Lower `alpha` 0.70→0.65, `max_hops` 2→3 for chain-depth DBs | +2 pp recall |
| **MEDIUM** | `CO_JOIN` edge weight 0.80→0.50 | better precision on stat-dense DBs |
| **LOW** | Expand synonym vocab (10/node via gpt-4o) | +1–2 pp recall |
| **FUTURE** | End-to-end EA evaluation (retrieval → SQL accuracy) | validation |

**v3 target:** GraphRetriever F1 ≥ 0.78, recall ≥ 0.95, precision ≥ 0.58, avg tokens ≤ 90

Current v2 baseline: F1 = 0.744, recall = 0.925, precision = 0.622, avg tokens = 77

---

## Key Decisions Made This Session

| Decision | Rationale |
|---|---|
| `retrieval/utils/` for npmi/linker/filter | These are HybridRetriever internals, not standalone retrievers — utils is the right label |
| `schema/__init__.py` re-exports everything | Any file at any depth uses `from src.schema import X` — no dot counting ever |
| `SchemaIndexer` added to schema `__init__` | It was the last remaining `..schema.X` relative import (in hybrid_retriever.py) |

---

## Test Status

**226 / 226 tests passing** after all changes.

---

## Important Notes

1. **`src/retrieval/utils/` relative imports use `...`** (3 dots) — `utils/` is one level deeper than before. Already fixed in this session; just be aware if adding new imports inside `retrieval/utils/`.

2. **`src/schema/__init__.py` imports `SchemaIndexer`** which depends on `chromadb`. If chromadb is not installed, importing anything from `src.schema` will fail. This was pre-existing behavior — `SchemaIndexer` always needed chromadb.

3. **Graph file not in repo** — `data/schema_graphs/bird_dev_enriched_v2.json` must be built locally with `build_schema_graph.py --enrich`.

4. **Two Chroma dirs in use** (from v2 benchmark session):
   - `./data/chroma_db_bird_analysis` — default for `analyze_retrieval.py`
   - `./data/chroma_db_bird_hybrid_only` — for parallel hybrid run

---

## Resources

- `CLAUDE.md` — full architecture, all commands, config flags
- `docs/handoff/handoff-20260317-phase3.md` — previous session (Phase 1–3 refactor)
- `docs/handoff/handoff-20260316-1118.md` — GraphRetriever v2/v3 algorithm context
- `docs/report.md` — full v2 BIRD Dev analysis (371 lines)
- PPR code: `src/schema_graph/graph_builder.py`
- Retriever: `src/retrieval/graph_retriever.py`

---

## To Continue

```
/continue handoff-20260317-phase4-structure.md
```
