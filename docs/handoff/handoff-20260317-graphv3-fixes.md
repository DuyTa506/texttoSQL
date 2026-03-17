# Handoff Document

**Created:** 2026-03-17
**Topic:** GraphRetriever v3 — audit + missing/broken feature fixes
**Status:** Completed — all fixes pushed, 226/226 tests passing
**Repo:** `https://github.com/DuyTa506/texttoSQL.git`
**Branch:** `main` (clean working tree, fully pushed)

---

## Context Summary

Audited the GraphRetriever v3 implementation against the spec in `CLAUDE.md`.
Found 4 bugs/missing items. Fixed all of them in one commit (`f684c89`).
Also completed a folder restructure session before this (`7e05fdf`, `066c96a`).

---

## Completed Work — This Session

### Folder restructure (previous commits — already in `handoff-20260317-phase4-structure.md`)
- `src/pre_retrieval/` — Phase 1 components moved out of `retrieval/`
- `src/retrieval/utils/` — NPMIScorer, BidirectionalLinker, SchemaFilter
- `src/schema/__init__.py` — populated, all deep relative imports flattened

### GraphRetriever v3 audit + fixes (`f684c89`)

| # | Issue | Fix | File |
|---|---|---|---|
| 1 | FK-column seed did not inject parent TABLE into PPR personalization vector | `_build_personalization()` now injects `tbl_nid` at `0.5 × col_score` for any FK column seed | `graph_builder.py` |
| 2 | `max_hops` parameter missing everywhere (CLAUDE.md listed it as a config flag) | Added `max_hops=2` to `GraphRetriever.__init__`, stored as `self.max_hops`, wired as `max_iter = max_hops * 50` in `SchemaGraph.retrieve()` call | `graph_retriever.py`, `default.yaml`, `run_pipeline.py` |
| 3 | `W_INFERRED_FK = 0.80` in code vs `0.95` in spec | Changed to `0.95` | `statistical_edges.py` |
| 4 | `_bidir(TABLE_FK)` for inferred table pairs emitted `TABLE_FK` for both A→B and B→A | Replaced with explicit `TABLE_FK` (A→B) + `TABLE_FK_REV` (B→A) | `statistical_edges.py` |

---

## Key Decisions Made

| Decision | Rationale |
|---|---|
| Keep `0.5 × col_score` for table seed weight (not `min(col_score, 0.5)`) | `min()` breaks the column > table hierarchy when `col_score < 0.5` — ratio becomes 1:1. Fixed ratio of 0.5 is semantically correct: table is always half as important as the matching column. After normalization, only ratios matter anyway. |
| `max_hops` maps to `max_iter = max_hops * 50` | PPR in networkx doesn't have a "hops" concept natively — it uses power iterations. `max_hops * 50` gives a reasonable convergence budget proportional to intended depth. |
| `W_INFERRED_FK = 0.95` (same as explicit FOREIGN_KEY) | Inferred FKs from high Jaccard value overlap (≥0.5, ≥3 shared values) are reliable — same weight as structural FK is appropriate. |

---

## Current v3 Status — All Items Done

| Feature | Status |
|---|---|
| TABLE_FK / TABLE_FK_REV edge types | ✅ |
| FK-column → TABLE seed injection in PPR | ✅ (fixed this session) |
| Per-DB adaptive `score_gap_ratio` (3 tiers) | ✅ |
| `alpha = 0.7` default | ✅ |
| `max_hops = 2` parameter | ✅ (fixed this session) |
| CO_JOIN weight cap 0.80 → 0.50 | ✅ |
| FK bridge injection (`use_fk_bridge`) | ✅ |
| INFERRED_FK / INFERRED_FK_REV defined and built | ✅ |
| `W_INFERRED_FK = 0.95` | ✅ (fixed this session) |
| `TABLE_FK_REV` correct direction in `_bidir` | ✅ (fixed this session) |

---

## Pending Work

### High priority — evaluate v3 gains
- [ ] Re-run `analyze_retrieval.py` on BIRD Dev to measure v3 vs v2 baseline
  - **v2 baseline:** F1 = 0.744, recall = 0.925, precision = 0.622, avg tokens = 77
  - **v3 target:** F1 ≥ 0.78, recall ≥ 0.95, precision ≥ 0.58, avg tokens ≤ 90
  - Need to rebuild graph with `build_schema_graph.py --enrich` first (graph not in repo)

### Medium priority — synonym vocab expansion
- [ ] Expand synonym vocab to 10 synonyms/node via gpt-4o (currently ~4–8)
  - Expected: +1–2 pp recall
  - Run `build_schema_graph.py --enrich --enrich_model gpt-4o`

### Deferred — wait for retrieval API stability
- [ ] `@dataclass RetrievalResult(id, chunk, score, source)` — replace raw `list[dict]`
- [ ] Decompose `HybridRetriever` into `BM25Retriever` + `SemanticRetriever`

---

## Files Modified This Session

```
src/schema_graph/graph_builder.py          ← FK-column TABLE seed injection
src/retrieval/graph_retriever.py           ← max_hops param added
src/schema_graph/edge_builders/
  statistical_edges.py                     ← W_INFERRED_FK 0.80→0.95, TABLE_FK_REV fix
configs/default.yaml                       ← schema_graph.max_hops: 2 added
scripts/run_pipeline.py                    ← max_hops wired from config
```

---

## Important Notes

1. **`_build_personalization` is no longer `@staticmethod`** — it now calls `self.nodes` to look up FK status. Any external code calling it as `SchemaGraph._build_personalization(scores, top_m)` must be updated to `instance._build_personalization(scores, top_m)`. (Only called internally via `self._build_personalization(...)` — no external callers.)

2. **Graph must be rebuilt** to benefit from `W_INFERRED_FK` fix and `TABLE_FK_REV` fix — these affect edge weights in the offline graph file. Existing `bird_dev_enriched_v2.json` uses old weights.

3. **`max_hops=2` → `max_iter=100`** — same as old `ppr_max_iter=100` default, so no behavior change at default settings. Increase `max_hops` to 3 for chain-depth DBs (per CLAUDE.md MEDIUM priority item).

---

## Resources

- `CLAUDE.md` — full architecture, all commands, config flags
- `docs/handoff/handoff-20260317-phase4-structure.md` — previous session
- `docs/report.md` — full v2 BIRD Dev analysis (371 lines)
- PPR code: `src/schema_graph/graph_builder.py` — `retrieve()` + `_build_personalization()`
- Retriever: `src/retrieval/graph_retriever.py`
- Edge builders: `src/schema_graph/edge_builders/statistical_edges.py`

---

## To Continue

```
/continue handoff-20260317-graphv3-fixes.md
```
