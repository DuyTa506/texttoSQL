# Research Paper Recommendation Plan

**Date:** 2026-03-17
**Project:** texttoSQL — Pre-Retrieval Query Understanding for Text-to-SQL
**Author:** Duy Ta

---

## 1. Landscape Analysis — Where We Stand

### 1.1 Current BIRD Leaderboard (as of March 2026)

| Rank | System | Size | Test EX (%) | Key Innovation |
|------|--------|------|-------------|----------------|
| 1 | AskData + GPT-4o (AT&T) | UNK | **81.95** | Industrial agentic pipeline |
| 2 | Agentar-Scale-SQL (Ant Group) | UNK | **81.67** | Multi-agent scaling |
| 3 | LongData-SQL (LongShine AI) | UNK | **77.53** | Long-context reasoning |
| 4 | CHASE-SQL + Gemini (Google) | UNK | **76.02** | Multi-candidate + self-consistency |
| ~15 | CHESS + GPT-4o (Stanford) | UNK | **71.10** | Schema selector + unit tester |
| ~20 | OmniSQL-32B (RMU + ByteDance) | 32B | **72.05** | Open-source SFT |
| ~30 | CSC-SQL + Qwen2.5-7B | 7B | **71.72** | 7B competitive with GPT-4 |
| ~40 | SLM-SQL + Qwen2.5-1.5B | 1.5B | **70.49** | 1.5B(!) near 70% |
| ~50 | Struct-SQL (Crater Labs) | **4B** | **60.42** | 4B baseline |
| — | DAIL-SQL + GPT-4 | UNK | 57.41 | Early ICL pioneer |
| — | MAC-SQL + GPT-4 | UNK | 59.59 | Multi-agent decompose |

### 1.2 Key Observations

1. **The gap between GPT-4 baseline (54.89%) and SOTA (81.95%) = 27pp** — most of this comes from better schema linking, value linking, and multi-agent orchestration, NOT from model capability alone.

2. **Small models are competitive**: SLM-SQL (1.5B) reaches 70.49%, CSC-SQL (7B) reaches 71.72% — within 10pp of SOTA. The bottleneck for SLMs is **retrieval quality**, not generation.

3. **No paper on the leaderboard focuses specifically on pre-retrieval as a standalone contribution.** All top systems bundle pre-retrieval into their agent pipeline. This is the gap.

4. **Schema graph + PPR for Text-to-SQL**: Not represented on the leaderboard. GNN-based approaches (Graphix-T5, SADGA, LGESQL) are older (2021-2022) and on Spider only. PPR-based typed heterogeneous graph is novel.

5. **CHESS (Stanford, 2024)** is the closest related work — it has a Schema Selector agent that prunes large schemas. But it's LLM-in-the-loop (expensive), not graph-based.

### 1.3 Published Work by Venue (2023-2025)

| Venue | Papers | Focus |
|-------|--------|-------|
| NeurIPS 2023 | BIRD benchmark (Spotlight) | Benchmark |
| VLDB 2024 | DAIL-SQL | ICL example selection |
| SIGMOD 2024 | CodeS (SFT baseline) | Fine-tuning |
| ACL 2024 Findings | TA-SQL | Schema alignment |
| ACL 2025 Main | SHARE | Self-correction via procedural steps |
| COLING 2025 | MAC-SQL (Oral) | Multi-agent decomposition |
| ICLR 2026 | BIRD-Interact (Oral) | Interactive benchmark |
| arXiv 2024 | CHESS, E-SQL, RSL-SQL, DTS-SQL | Various |

---

## 2. Research Gap Analysis — What's Missing

### 2.1 Gaps We Can Exploit

| Gap | Evidence | Our Advantage |
|-----|----------|---------------|
| **No standalone pre-retrieval study** | All top systems bundle schema linking into monolithic agents. No ablation of pre-retrieval components in isolation. | We have modular, feature-flagged pre-retrieval components (ValueScanner, Decomposer, Augmentor) already built and measurable independently. |
| **No PPR on typed heterogeneous schema graph** | GNN approaches (SADGA, LGESQL) learn on graph. PPR with hand-crafted typed edges (3 layers) is unexplored for Text-to-SQL. | Our SchemaGraph with 14 edge types + PPR + adaptive pruning is fully implemented. |
| **No numeric/temporal value linking** | CHESS does string value matching. No system handles `"above 3.5"` → `col > 3.5` or `"Q1 2023"` → date range. BIRD has heavy numeric conditions. | ValueScanner framework extensible to numeric. |
| **No adaptive retrieval routing** | All systems use fixed retrieval strategy. No complexity-based routing. | QuestionDecomposer already has complexity scoring. |
| **SLM + good retrieval underexplored** | Most leaderboard entries use GPT-4o or 32B+ models. Few study what happens when you give a 4B model excellent schema context. | Our pipeline: Qwen3-4B + GraphRetriever v3. |

### 2.2 Gaps We CANNOT Exploit (avoid these)

| Gap | Why Not |
|-----|---------|
| Multi-turn / conversational SQL | Different problem (BIRD-Interact is a separate benchmark) |
| End-to-end GNN training on schema graph | Too expensive, need GPU cluster, competing with well-funded labs |
| Cross-database transfer learning | Out of scope for a retrieval-focused paper |

---

## 3. Proposed Paper — Two Viable Stories

### Story A: "Schema Knowledge Graph with Pre-Retrieval Boosting for Small Language Model Text-to-SQL" (Recommended)

**Target venues:** EMNLP 2025 (deadline ~Jun 2026), ACL 2026 (deadline ~Oct 2025, likely passed), COLING 2026, or NAACL 2026

**Core thesis:** A typed heterogeneous schema knowledge graph with Personalized PageRank retrieval, combined with question-adaptive pre-retrieval boosting (value linking, schema mention injection, numeric normalization), enables a 4B parameter model to match or exceed GPT-4 + basic prompting on BIRD Dev.

**Novelty claims:**
1. **Typed 3-layer schema knowledge graph** (structural + LLM-enriched semantic + statistical edges) with PPR — first application to Text-to-SQL
2. **Adaptive pre-retrieval router** — complexity-driven strategy selection (keyword / value / decompose)
3. **Numeric + temporal value linking** — extends string-only value scanning to handle quantitative conditions
4. **Schema mention pre-linking** — zero-latency exact-match seed injection into PPR personalization
5. **SLM-competitive results** — Qwen3-4B + our retrieval vs. GPT-4 + vanilla prompting

**Why this works as a paper:**
- Clear modular ablation story (each component adds measurable pp)
- Novel combination (graph + pre-retrieval + SLM) not on any leaderboard
- Practical: entire pipeline runs on consumer GPU
- Reproducible: all code, all configs, open-source model

---

### Story B: "Question-Adaptive Pre-Retrieval for Schema Linking in Text-to-SQL" (Narrower, faster to write)

**Target venues:** Workshop papers (NLP4DB @ VLDB, SemEval, or EMNLP Findings)

**Core thesis:** Pre-retrieval question understanding (value scanning, decomposition, augmentation) is systematically underexplored. We provide the first standalone ablation study.

**Advantage:** Faster to write, less implementation needed
**Disadvantage:** Smaller contribution, findings-level at best

---

## 4. Recommended: Story A — Detailed Implementation Plan

### 4.1 What's Already Done ✅

| Component | Status | Notes |
|-----------|--------|-------|
| SchemaGraph with 14 edge types | ✅ Done | 3-layer edges, PPR retrieval |
| GraphRetriever v3 | ✅ Done | FK bridge, adaptive pruning, synonym boost |
| ValueScanner (string fuzzy) | ✅ Done | TEXT columns only |
| QuestionDecomposer | ✅ Done | Rule-based complexity scoring |
| QueryAugmentor (3 strategies) | ✅ Done | keyword / value / decompose |
| LLM Node Enrichment | ✅ Done | GPT-4o-mini batch enrichment |
| HybridRetriever baseline | ✅ Done | BM25 + semantic + NPMI |
| Qwen3-4B SFT training pipeline | ✅ Done | LoRA + GRPO RL |
| analyze_retrieval.py | ✅ Done | Retrieval metrics on BIRD Dev |
| BIRD Dev graph (v3 rebuild) | ✅ Done | 884 nodes, 6981 edges |

### 4.2 What Needs to Be Built 🔨

#### Week 1: Pre-Retrieval Extensions (~3 days)

| Task | File | Effort | Impact |
|------|------|--------|--------|
| **Schema mention pre-linking** | `graph_retriever.py` → `_build_personalization()` | ~30 lines, 2 hrs | +1-2 pp recall |
| **Numeric value extractor** | `value_scanner.py` → new `_extract_numeric_mentions()` | ~100 lines, 4 hrs | +2-4 pp on BIRD |
| **Temporal mention detector** | `value_scanner.py` → new `_extract_temporal_mentions()` | ~80 lines, 3 hrs | +1-2 pp on BIRD |
| **Adaptive pre-retrieval router** | `pre_retrieval/router.py` (new) | ~150 lines, 4 hrs | Framework contribution |
| **Wire all into `run_pipeline.py`** | `scripts/run_pipeline.py` | ~50 lines, 2 hrs | — |

**Schema Mention Pre-Linking** (highest priority):
```python
# In GraphRetriever._build_personalization() or GraphRetriever.retrieve()
def _inject_schema_mentions(self, question: str, scores: dict) -> dict:
    """Zero-latency: boost nodes whose names appear as substrings in question."""
    q_tokens = set(re.findall(r'[a-zA-Z_]\w+', question.lower()))
    for node_id, node in self.graph.nodes.items():
        name_tokens = set(node.table_name.lower().split('_'))
        if node.column_name:
            name_tokens |= set(node.column_name.lower().split('_'))
        overlap = q_tokens & name_tokens
        if overlap and len(overlap) >= 1:
            boost = 0.8 if len(overlap) >= 2 else 0.5
            scores[node_id] = max(scores.get(node_id, 0), boost)
    return scores
```

**Numeric Value Extractor**:
```python
def _extract_numeric_mentions(self, question: str) -> list[ValueMatch]:
    """Extract numeric values and match to numeric DB columns."""
    patterns = [
        (r'(?:above|over|more than|greater than|>)\s*(\d+\.?\d*)', 'GT'),
        (r'(?:below|under|less than|fewer than|<)\s*(\d+\.?\d*)', 'LT'),
        (r'(?:at least|>=|minimum)\s*(\d+\.?\d*)', 'GTE'),
        (r'(?:at most|<=|maximum)\s*(\d+\.?\d*)', 'LTE'),
        (r'(?:exactly|equal to|=)\s*(\d+\.?\d*)', 'EQ'),
        (r'(?:between)\s*(\d+\.?\d*)\s*(?:and)\s*(\d+\.?\d*)', 'BETWEEN'),
        (r'\b((?:19|20)\d{2})\b', 'YEAR'),  # years like 2023, 1995
    ]
    # Match extracted numbers to numeric columns by checking min/max ranges
    ...
```

**Adaptive Pre-Retrieval Router**:
```python
class PreRetrievalRouter:
    """Route questions to optimal pre-retrieval strategy based on complexity."""

    def classify(self, question: str, db: Database) -> str:
        """Returns: 'simple' | 'value_dense' | 'multi_hop' | 'schema_heavy'"""
        features = {
            'has_values': bool(re.search(r'["\']|(?:19|20)\d{2}|\d+\.\d+', question)),
            'has_comparisons': bool(re.search(r'compare|versus|vs|difference|between', question, re.I)),
            'has_schema_mentions': self._count_schema_mentions(question, db) >= 2,
            'complexity_score': QuestionDecomposer().complexity_score(question),
            'token_count': len(question.split()),
        }

        if features['has_schema_mentions']:
            return 'schema_heavy'  # → direct seed injection, skip augmentation
        elif features['has_values']:
            return 'value_dense'   # → value scan + numeric extraction
        elif features['has_comparisons'] or features['complexity_score'] > 0.6:
            return 'multi_hop'     # → decompose + multi-query retrieval
        else:
            return 'simple'        # → keyword augmentation only
```

#### Week 2: Full Ablation Experiments (~4 days)

| Experiment | Config | Metrics | Purpose |
|------------|--------|---------|---------|
| **E1: Baseline** | HybridRetriever, no pre-retrieval | recall, precision, F1, EX | Baseline |
| **E2: GraphRetriever v3** | Graph only, no pre-retrieval | recall, precision, F1, EX | Graph contribution |
| **E3: + Schema mention pre-linking** | Graph + mention boost | recall, precision, F1 | Component ablation |
| **E4: + Value scanning (string)** | Graph + ValueScanner | recall, precision, F1 | Component ablation |
| **E5: + Numeric value linking** | Graph + ValueScanner + numeric | recall, precision, F1 | Novel component |
| **E6: + Adaptive router** | Graph + router (auto strategy) | recall, precision, F1 | Framework contribution |
| **E7: Full pipeline** | Graph + all pre-retrieval + router | recall, precision, F1, EX | Full system |
| **E8: Graph + Hybrid merge** | Graph + HybridRetriever via RRF | recall, precision, F1 | Hybrid benefit |
| **E9: Cross-retriever** | Pre-retrieval + HybridRetriever (no graph) | recall, precision, F1 | Generalization |

**Datasets:**
- **BIRD Dev** (1,534 questions) — primary
- **Spider Dev** (1,034 questions) — cross-benchmark
- **BIRD Mini-Dev** (500 questions, 3 dialects) — for quick iteration

**Run commands:**
```bash
# E1: Baseline
uv run python scripts/analyze_retrieval.py --data_path datasets/data/bird/dev_20240627 \
    --retriever hybrid --num_examples 0

# E2: GraphRetriever v3
uv run python scripts/analyze_retrieval.py --data_path datasets/data/bird/dev_20240627 \
    --retriever graph --graph_path data/schema_graphs/bird_dev_v3_rebuild.json \
    --use_fk_bridge --score_gap_ratio 3.0 --alpha 0.7 --top_m 5

# E3-E7: Need new CLI flags for each component on/off
# Will add: --schema_mention, --value_scan, --numeric_value, --router
```

#### Week 3: End-to-End Evaluation + Paper Writing (~5 days)

| Task | Effort | Output |
|------|--------|--------|
| Run Qwen3-4B on BIRD Dev with E7 schema context | 2 hrs (GPU) | EX accuracy |
| Run GPT-4o-mini with E7 vs E1 | 1 hr ($5-10) | Comparison to API model |
| Error analysis: categorize 50 failure cases | 4 hrs | Table in paper |
| Write paper (8 pages) | 3 days | Full draft |
| Generate figures (architecture diagram, ablation chart, pareto curve) | 4 hrs | Figures 1-4 |

### 4.3 Paper Structure

```
Title: "Schema Knowledge Graph with Adaptive Pre-Retrieval
        for Small Language Model Text-to-SQL"

Abstract: 250 words

1. Introduction (1 page)
   - Problem: Schema linking bottlenecks SLMs on BIRD
   - Gap: No systematic pre-retrieval study; no PPR on typed graph for NL2SQL
   - Contribution: (1) 3-layer typed graph + PPR, (2) adaptive pre-retrieval,
                   (3) numeric value linking, (4) SLM-competitive results

2. Related Work (1 page)
   - Schema linking: SADGA, LGESQL, Graphix-T5 (GNN-based),
                     CHESS (LLM-based), DAIL-SQL (ICL)
   - Value linking: CHESS IR agent, TA-SQL alignment
   - Pre-retrieval: QueryAugmentor patterns (keyword, decompose)
   - Small models: CodeS, OmniSQL, SLM-SQL

3. Method (2.5 pages)
   3.1 Schema Knowledge Graph Construction
       - Node types (DATABASE, TABLE, COLUMN)
       - Layer 1: Structural edges (FK, PK, membership)
       - Layer 2: Semantic edges (embedding, synonym, lexical)
       - Layer 3: Statistical edges (co-occurrence, value overlap)
       - LLM enrichment (descriptions + synonyms)
   3.2 PPR-based Schema Retrieval
       - Seed scoring (embedding cosine + synonym boost)
       - Personalized PageRank (alpha=0.7, max_hops=2)
       - FK bridge injection
       - Adaptive score-gap pruning (3-tier)
   3.3 Adaptive Pre-Retrieval Framework
       - Complexity classifier (rule-based)
       - Schema mention pre-linking
       - Numeric + temporal value extraction
       - Strategy routing (simple / value_dense / multi_hop / schema_heavy)
   3.4 Integration with SLM Generation
       - SchemaFilter → CREATE TABLE prompt
       - Qwen3-4B with SFT + GRPO

4. Experiments (2 pages)
   4.1 Setup (datasets, metrics, baselines)
   4.2 Retrieval Quality (Table 1: ablation on recall/precision/F1)
   4.3 End-to-End Accuracy (Table 2: EX on BIRD Dev, Spider Dev)
   4.4 Comparison with Existing Systems (Table 3)
   4.5 SLM vs. LLM with Our Retrieval (Table 4)

5. Analysis (1 page)
   5.1 Error Analysis (Figure 3: error categories)
   5.2 Latency Analysis (Table 5: ms per component)
   5.3 Cross-Retriever Generalization (does pre-retrieval help HybridRetriever too?)
   5.4 When Does the Router Help? (per-category breakdown)

6. Conclusion (0.5 page)

References (~40 citations)
Appendix: Full config, more examples, per-DB breakdown
```

### 4.4 Expected Results (Realistic Estimates)

| Config | BIRD Dev Recall | BIRD Dev F1 | BIRD Dev EX (Qwen3-4B) |
|--------|----------------|-------------|------------------------|
| E1: HybridRetriever baseline | 0.72 | 0.65 | ~45% |
| E2: GraphRetriever v3 | 0.88 | 0.84 | ~50% |
| E3: + Schema mention | 0.90 | 0.85 | ~51% |
| E5: + Numeric value | 0.92 | 0.87 | ~53% |
| E7: Full pipeline | 0.93 | 0.88 | ~55% |
| GPT-4 + vanilla prompting | — | — | ~55% |

**Key claim:** Qwen3-4B + our retrieval ≈ GPT-4 + vanilla, at fraction of the cost.

### 4.5 Comparison Positioning

| System | Model | BIRD Test EX | Our Position |
|--------|-------|-------------|--------------|
| ChatGPT + CoT | GPT-3.5 | 40.08% | We beat this easily |
| GPT-4 (baseline) | GPT-4 | 54.89% | We target this range with 4B model |
| DAIL-SQL + GPT-4 | GPT-4 | 57.41% | Competitive |
| MAC-SQL + GPT-4 | GPT-4 | 59.59% | Aspirational target |
| Struct-SQL | **4B** | **60.42%** | Direct competitor (same model size) |
| SLM-SQL + 1.5B | 1.5B | 70.49% | They use Oracle Knowledge; we don't |

**Note:** Many top systems use "Oracle Knowledge" (BIRD's external knowledge hints). If we also use Oracle Knowledge, our numbers would be higher. Without it, ~55% EX with a 4B model would be a strong result.

---

## 5. Timeline

```
Week 1 (Days 1-3): Implementation
  ├─ Day 1: Schema mention pre-linking + numeric value extractor
  ├─ Day 2: Temporal mention detector + adaptive router
  └─ Day 3: Wire into pipeline + unit tests

Week 2 (Days 4-7): Experiments
  ├─ Day 4: Run E1-E5 retrieval ablations (BIRD Dev)
  ├─ Day 5: Run E6-E9 retrieval ablations + Spider Dev
  ├─ Day 6: Run end-to-end EX with Qwen3-4B (if GPU available)
  └─ Day 7: Error analysis (50 failure cases)

Week 3 (Days 8-12): Writing
  ├─ Days 8-9: Method section (Section 3)
  ├─ Day 10: Experiments + Analysis (Sections 4-5)
  ├─ Day 11: Intro + Related Work + Conclusion
  └─ Day 12: Figures, tables, polish, submit to arXiv

Total: ~12 working days (~2.5 weeks)
```

---

## 6. Venue Recommendations

| Venue | Deadline (estimated) | Fit | Notes |
|-------|---------------------|-----|-------|
| **EMNLP 2026** | ~Jun 2026 | ⭐⭐⭐⭐ | Best fit: NLP + DB, findings track realistic |
| **NAACL 2026** | ~Jan 2026 (passed?) | ⭐⭐⭐⭐ | Similar fit |
| **COLING 2026** | TBD | ⭐⭐⭐ | MAC-SQL published here |
| **NeurIPS 2026 DB Track** | ~May 2026 | ⭐⭐⭐ | Competitive but high visibility |
| **ACL 2026 Findings** | ~Feb 2026 | ⭐⭐⭐⭐ | TA-SQL was here, good precedent |
| **VLDB 2027** | Rolling | ⭐⭐⭐ | DB venue, care about scalability |
| **arXiv first** | Anytime | — | Establish priority, submit to venue later |

**Recommendation:** Submit to arXiv immediately after completing experiments. Then target EMNLP 2026 or COLING 2026.

---

## 7. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Retrieval gains don't translate to EX accuracy | Medium | Focus paper on retrieval metrics (recall/F1); EX is secondary |
| Qwen3-4B too weak for BIRD complexity | Medium | Also report GPT-4o-mini results as upper bound |
| Reviewers say "just engineering, no novelty" | Low-Medium | Emphasize: (1) first PPR on typed graph for NL2SQL, (2) first systematic pre-retrieval ablation, (3) numeric value linking is novel |
| Similar work appears before submission | Low | PPR + pre-retrieval combination is niche; submit to arXiv fast |
| BIRD Dev numbers lower than expected | Medium | Report relative improvement (Δ) not absolute; v2→v3 was already measured |

---

## 8. Deliverables Checklist

### Code
- [ ] `src/pre_retrieval/router.py` — Adaptive pre-retrieval router
- [ ] `src/pre_retrieval/value_scanner.py` — Extended with numeric + temporal
- [ ] `src/retrieval/graph_retriever.py` — Schema mention pre-linking
- [ ] `scripts/analyze_retrieval.py` — New CLI flags for ablation
- [ ] `configs/paper_ablations/` — 9 config files (E1-E9)
- [ ] Tests for all new components

### Experiments
- [ ] Table 1: Retrieval ablation (9 configs × 2 datasets)
- [ ] Table 2: End-to-end EX accuracy
- [ ] Table 3: Comparison with published systems
- [ ] Table 4: SLM vs LLM with our retrieval
- [ ] Table 5: Latency breakdown
- [ ] Figure 1: Architecture diagram
- [ ] Figure 2: Ablation bar chart
- [ ] Figure 3: Error analysis pie chart
- [ ] Figure 4: Pareto curve (latency vs accuracy)

### Paper
- [ ] 8-page main paper (ACL format)
- [ ] Appendix (unlimited)
- [ ] Camera-ready code release on GitHub

---

## 9. Summary

**The strongest paper story combines your three unique assets:**

1. **Schema Knowledge Graph** (typed, 3-layer, PPR) — novel for Text-to-SQL
2. **Adaptive Pre-Retrieval** (router + numeric value linking) — systematically unexplored
3. **SLM-competitive results** (4B model matching GPT-4 baseline) — practical impact

**This combination is not on any leaderboard and not in any published paper.** The implementation gap is ~3 days of coding + ~4 days of experiments + ~5 days of writing = **12 working days total**.
