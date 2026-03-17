# Handoff Document

**Created:** 2026-03-17
**Topic:** Full `src/` refactoring — Phase 1 → 2 → 3 (safe items)
**Status:** Completed — all safe refactoring done, ready for v3 algorithm work
**Repo:** `https://github.com/DuyTa506/texttoSQL.git`
**Branch:** `main` (all pushed, clean working tree)

---

## Context Summary

Three full refactoring phases were applied to `src/` (training code untouched).
No algorithm or behavior changes — all 226 tests pass after every commit.
Four commits pushed this session on top of the existing GraphRetriever v3 feature work.

---

## Completed Work — This Session

### Phase 1 Quick Wins (`0b2e381`)
- [x] `value_scanner.py` — regex patterns & thresholds → module-level constants; inline `import re` removed from `_extract_candidates`; `conn.close()` → `with sqlite3.connect() as conn:`
- [x] `question_decomposer.py` — all complexity weights → class-level constants (`_LEN_DIVISOR`, `_KEYWORD_CONTRIB`, etc.)
- [x] `retry_loop.py` — inline `120` → `_ERROR_MSG_PREVIEW_LEN` constant
- [x] `query_augmentor.py` — added missing return type `str | list[str]` on `augment()`
- [x] `node_enricher.py` — replaced direct `SentenceTransformer` with `BaseEmbeddingModel` abstraction; `embed_nodes()` accepts `BaseEmbeddingModel | str`
- [x] All 28 `src/` files — `Optional[X]` → `X | None`, `List[X]` → `list[X]`, `Dict` → `dict`

### Phase 1 Bonus — Retrieval Architecture (`52fc4a0`)
- [x] `src/retrieval/base_retriever.py` [NEW] — `BaseRetriever(ABC)` with abstract `retrieve()` and `retrieve_multi()`, both accepting `value_matches`
- [x] `GraphRetriever` moved: `src/schema_graph/graph_retriever.py` → `src/retrieval/graph_retriever.py`
- [x] `HybridRetriever` and `GraphRetriever` both inherit `BaseRetriever`
- [x] `src/retrieval/__init__.py` — exports all three from one package
- [x] `scripts/run_pipeline.py` + `analyze_retrieval.py` — import paths updated

### Phase 2 (`b81edc4`)
- [x] `src/shared/sqlite_utils.py` [NEW] — `safe_execute(db_path, sql) → (rows, err)` context-manager helper replacing 3 duplicate raw sqlite3 blocks
- [x] `evaluation/metrics.py` — `execution_accuracy()` uses `safe_execute`
- [x] `post/sql_executor.py` — `execute()` + `_run_gold()` use `safe_execute`; `_classify_operational_error` merged/renamed `_classify_error`; `import sqlite3` removed
- [x] `schema_graph/graph_builder.py` — `SchemaGraph.retrieve()` (~180 lines) split into 5 private helpers: `_get_entry_points()`, `_merge_value_boosts()`, `_build_personalization()`, `_run_ppr()`, `_filter_and_sort()`
- [x] `retrieval/schema_filter.py` — `filter_and_format()` (~127 lines) split into 6 private helpers: `_select_top_chunks()`, `_group_by_table()`, `_collect_col_scores()`, `_cols_to_show()`, `_format_table_block()`, `_format_fk_summary()`, `_inject_value_hints()`
- [x] `src/llms/chat_format.py` [NEW] — `messages_to_chatml()`, `build_chatml()`, `DEFAULT_SYSTEM_PROMPT`
- [x] `generation/inference.py` — `_messages_to_chatml()` delegates to `chat_format`
- [x] `llms/huggingface.py` — local `_build_chatml` + constant replaced with imports from `chat_format`

### Phase 3 Safe (`e45c415`)
- [x] `src/llms/factory.py` — `LLMConfig` dataclass with `from_dict()` + `to_kwargs()`; `LLMFactory.from_config()` is now a 3-line wrapper; `LLMConfig` exported from `llms/__init__.py`
- [x] `src/evaluation/benchmark.py` — `BenchmarkReporter` class extracted (owns `display()` + `save()`); `Benchmark` injects reporter (defaults to `BenchmarkReporter()` — no breaking change); module-level `console` singleton removed
- [x] `src/evaluation/__init__.py` [NEW] — proper package entry point exporting `Benchmark`, `BenchmarkReporter`, and all metrics functions

---

## Current `src/` Structure (new files only)

```
src/
  shared/
    __init__.py
    sqlite_utils.py         ← safe_execute() helper [NEW]
  llms/
    chat_format.py          ← canonical ChatML builder [NEW]
    factory.py              ← + LLMConfig dataclass
    __init__.py             ← + LLMConfig export
  retrieval/
    base_retriever.py       ← BaseRetriever(ABC) [NEW]
    graph_retriever.py      ← MOVED from schema_graph/
    __init__.py             ← exports all three retrievers
  evaluation/
    __init__.py             ← proper package entry point [NEW]
    benchmark.py            ← + BenchmarkReporter class
```

---

## Pending Work — Phase 3 Deferred (wait for API stability)

These were explicitly deferred until GraphRetriever v3 algorithm is stable:

- [ ] `@dataclass RetrievalResult(id, chunk, score, source)` — replace raw `list[dict]` across all retrievers + callers. **High risk**, touches every retriever, schema_filter, run_pipeline, benchmark, tests.
- [ ] Decompose `HybridRetriever` into composed `BM25Retriever` + `SemanticRetriever`. **High risk**, same reason.

---

## Pending Work — GraphRetriever v3 Algorithm (highest priority)

From previous handoff `handoff-20260316-1118.md` — still open, **this is the next thing to work on**:

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
| `BaseRetriever` ABC includes `value_matches` | Uniform call site in `run_pipeline.py` — no `isinstance` checks needed |
| `HybridRetriever.retrieve()` accepts but ignores `value_matches` | Interface compat today; easy to add value-hint support later |
| Phase 3 A+B (RetrievalResult, HybridRetriever decompose) deferred | Retrieval API still evolving — v3 algorithm changes may require interface adjustments |
| `BenchmarkReporter` injected (not subclassed) | Composition over inheritance — simpler to swap for tests or custom reporters |
| `LLMConfig.extra` collects all unknown kwargs | Forward-compatible — new provider kwargs need no factory changes |

---

## Test Status

**226 / 226 tests passing** after all changes.

---

## Important Notes

1. **`src/schema_graph/graph_retriever.py` is deleted** — import path is now `src.retrieval.graph_retriever`. Any external scripts that import from the old path will break.

2. **`value_scanner.py` still has `from typing import Optional`** at top — was missed by the sweep. Not critical but can be cleaned up.

3. **Graph file not in repo** — `data/schema_graphs/bird_dev_enriched_v2.json` must be built locally with `build_schema_graph.py --enrich`.

4. **Two Chroma dirs in use** (from v2 benchmark session):
   - `./data/chroma_db_bird_analysis` — default for `analyze_retrieval.py`
   - `./data/chroma_db_bird_hybrid_only` — for parallel hybrid run

---

## Resources

- `CLAUDE.md` — full architecture, all commands, config flags
- `docs/handoff/handoff-20260316-1118.md` — GraphRetriever v2/v3 algorithm context (failure analysis, per-DB breakdown, top missed tables)
- `docs/report.md` — full v2 BIRD Dev analysis (371 lines)
- `configs/default.yaml` — feature flags
- PPR code: `src/schema_graph/graph_builder.py` — `retrieve()` + new helpers
- Retriever: `src/retrieval/graph_retriever.py` (~586 lines)

---

## To Continue

```
/continue handoff-20260317-phase3.md
```
