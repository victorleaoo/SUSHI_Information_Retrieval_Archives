# Phase 1: Query Expansion — Implementation Walkthrough

## What Was Built

This document describes everything added or modified to support Phase 1 (Set 1A, 1B, 1B-R) experiments.

---

## File Map

### `src/utils/` — Core Utilities

| File | Status | Purpose |
|------|--------|---------|
| [llm_client.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/utils/llm_client.py) | **Updated** | Added **Ollama** provider (`qwen3:12b` default). Added `logs_dir` param: every LLM call (cached or live) writes a JSON log. |
| [snc_classifier.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/utils/snc_classifier.py) | **NEW** | `classify_snc_specificity(snc)` → `'specific'` or `'generic'`. Used in Phase 2 prompts. |
| [field_resolver.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/utils/field_resolver.py) | Unchanged | Resolves folder metadata fields to prompt variables. |

### `src/query_expansion/` — Query Generators

| File | Status | Query Type | Output File |
|------|--------|-----------|-------------|
| [generate_hyde.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/query_expansion/generate_hyde.py) | Updated | Q1 — HyDE (SNC-style fake labels) | `data/queries_hyde.json` |
| [generate_keywords.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/query_expansion/generate_keywords.py) | Updated | Q2 — BM25 Keywords (entities/people) | `data/queries_keywords.json` |
| [generate_emb_text.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/query_expansion/generate_emb_text.py) | Updated | Q3 — Embedding Text (descriptive para) | `data/queries_embtext.json` |
| [generate_reformulations.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/query_expansion/generate_reformulations.py) | Updated | Q4 — 4 reformulations from 4 angles | `data/queries_reformulations.json` |
| [generate_concept.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/query_expansion/generate_concept.py) | **NEW** | Q5 — Concept Bridge (3-level categories) | `data/queries_concept.json` |
| [generate_hyde_r.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/query_expansion/generate_hyde_r.py) | **NEW** | Q6 — SNC-Aware HyDE (real label style) | `data/queries_hyde_r.json` |
| [generate_thematic.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/query_expansion/generate_thematic.py) | **NEW** | Q7 — Thematic Paragraph (concept-focused) | `data/queries_thematic.json` |
| [generate_keywords_r.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/query_expansion/generate_keywords_r.py) | **NEW** | Q2R — Term-Focused Keywords (no proper nouns) | `data/queries_keywords_r.json` |
| [combine_queries.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/query_expansion/combine_queries.py) | Unchanged | Builds Q0+Qx, QALL, Q0+QALL | Multiple `data/queries_*.json` |
| [combine_queries_r.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/query_expansion/combine_queries_r.py) | **NEW** | Builds Q0+Q5, Q0+Q6, Q0+Q5+Q6, Q0+QALL-R | Multiple `data/queries_*.json` |

> **Updates to existing generators:** All four original generators were updated to:
> - Accept a `logs_path` keyword argument (JSONL log per query type)
> - Skip already-cached topics (incremental generation)
> - Use consistent `[Q1] → topic_id` print format

### `scripts/` — Experiment Runners

| File | Status | Purpose |
|------|--------|---------|
| [run_phase1_generate_queries.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/scripts/run_phase1_generate_queries.py) | **NEW** | Runs all 8 query generators + both combiners. Use `--provider` and `--model` flags. |
| [run_phase1_1a.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/scripts/run_phase1_1a.py) | **NEW** | Runs 28 experiments (F1–F4 × B/E/C/BCE/BC/BE/CE). Saves `f_best.json`. |
| [run_phase1_1b.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/scripts/run_phase1_1b.py) | **NEW** | Runs 40 experiments with original expanded queries (Q1–Q4 + combos). |
| [run_phase1_1br.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/scripts/run_phase1_1br.py) | **NEW** | Runs 19 experiments with revised concept-focused queries (Q5–Q2R + combos). |
| [report_phase1.py](file:///Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/scripts/report_phase1.py) | **NEW** | Generates `results/phase1_report.md` comparing all 87 runs. |

---

## How to Run (Step by Step)

> **Important:** All scripts must be run from the **repository root** (`SUSHI_Information_Retrieval_Archives/`), not from `scripts/`.

### Step 0 — Start Ollama

Make sure the Ollama server is running with the `qwen3:12b` model:

```bash
ollama serve            # start server (if not running as a service)
ollama pull qwen3:12b   # download model (first time only)
```

### Step 1 — Generate All Queries

```bash
cd /Users/victorleao/mestrado/SUSHI_Information_Retrieval_Archives
python scripts/run_phase1_generate_queries.py
```

- Uses Ollama with `qwen3:12b` by default
- Safe to re-run — already-cached topics are skipped instantly
- ~360 LLM calls total (45 topics × 8 query types)
- **Estimated time:** depends on Ollama speed (~2–5 min/topic for qwen3:12b)

Override provider:
```bash
python scripts/run_phase1_generate_queries.py --provider groq
python scripts/run_phase1_generate_queries.py --provider ollama --model llama3
```

### Step 2 — Run Set 1A (28 experiments)

```bash
python scripts/run_phase1_1a.py
# Resume if interrupted:
python scripts/run_phase1_1a.py --skip-existing
```

Saves results to `all_runs/Phase1/1A/` and writes `f_best.json` automatically.

### Step 3 — Run Set 1B (40 experiments)

```bash
python scripts/run_phase1_1b.py
python scripts/run_phase1_1b.py --skip-existing   # to resume
```

Reads `f_best.json` and all `data/queries_*.json` files.

### Step 4 — Run Set 1B-R (19 experiments)

```bash
python scripts/run_phase1_1br.py
python scripts/run_phase1_1br.py --skip-existing  # to resume
```

### Step 5 — Generate the Results Report

```bash
python scripts/report_phase1.py
```

Opens `results/phase1_report.md` — contains all nDCG@5 tables with Δ deltas.

---

## Logging & Monitoring

When running in the background, all progress is written to:

| Log Location | Contents |
|--------------|----------|
| `data/logs/llm_calls/` | Per-call JSON: timestamp, provider, model, prompt, response, cached |
| `data/logs/phase1_queries/q1_hyde.jsonl` | Per-topic JSONL: prompt → processed query for Q1 |
| `data/logs/phase1_queries/q5_concept.jsonl` | Same for Q5 |
| *(etc. for all 8 query types)* | |
| `data/logs/phase1_runs/1a_run_log.jsonl` | Per-run JSONL: run_id, config, model, nDCG@5, duration |
| `data/logs/phase1_runs/1b_run_log.jsonl` | Same for 1B |
| `data/logs/phase1_runs/1br_run_log.jsonl` | Same for 1BR |

**To monitor a background run:**
```bash
# Watch 1A progress (last 5 entries)
tail -f data/logs/phase1_runs/1a_run_log.jsonl | python -c "import sys,json; [print(json.dumps(json.loads(l), indent=2)) for l in sys.stdin]"

# Quick status — see all completed run IDs and scores
cat data/logs/phase1_runs/1a_run_log.jsonl | python -c "
import sys, json
for line in sys.stdin:
    e = json.loads(line)
    print(f\"{e['run_id']:8s}  {e['config']}  {e['model']:4s}  nDCG@5={e['ndcg_at_5']:.4f}  ({e['duration_s']:.0f}s)\")
"
```

---

## Output Structure

```
all_runs/Phase1/
├── 1A/
│   ├── f_best.json                              ← F_best per model
│   ├── 1A-01_F1_B/
│   │   ├── run.txt                              ← TREC run file
│   │   ├── AllDocuments_TopicsFolderMetrics.json
│   │   ├── all_documents_model_overall_stats.json  ← nDCG@5 here
│   │   └── topics_mean_margin.json
│   └── ... (27 more run folders)
├── 1B/
│   └── ... (40 run folders)
└── 1BR/
    └── ... (19 run folders)

data/
├── queries_hyde.json           ← Q1: 45 topics
├── queries_keywords.json       ← Q2
├── queries_embtext.json        ← Q3
├── queries_reformulations.json ← Q4
├── queries_concept.json        ← Q5 (NEW)
├── queries_hyde_r.json         ← Q6 (NEW)
├── queries_thematic.json       ← Q7 (NEW)
├── queries_keywords_r.json     ← Q2R (NEW)
├── queries_q0q*.json           ← Combined queries
├── queries_qall*.json          ← QALL, QALL-R
├── queries_overview.json       ← Human-readable inspection (original)
├── queries_overview_r.json     ← Human-readable inspection (revised)
└── llm_cache/                  ← SHA256-keyed response cache

results/
└── phase1_report.md            ← Generated after all runs complete
```
