# Experiment Workbook: Folder Ranking in Sparsely Digitized Archives
**SUSHI Collection — Victor Hugo Oliveira Leão, 2026**

> This document is a living reference. Fill in decisions and results as experiments progress. Every `☐` is a task; every `___` is a value to fill after running experiments.

---

## Reference Guide

### Retrieval Models
| Code | Model | Notes |
|------|-------|-------|
| B | BM25F | Tuned. For AllF phase, retune for folder fields (not TOFS). Default: k1=1.5, b=0.75 |
| E | Sentence Embeddings (all-mpnet-base-v2) | Dense bi-encoder |
| C | ColBERTv2 (PLAID) | Late interaction |
| BCE | RRF(B + C + E, k=60) | Primary ensemble |
| BC | RRF(B + C) | — |
| BE | RRF(B + E) | — |
| CE | RRF(C + E) | — |

### Masking Conditions
| Code | Description | ECFs | Notes |
|------|-------------|------|-------|
| — | Phase 1 (AllF only, no docs used) | 1 | Evaluated once on 45 topics. No ECF variation |
| Uni | Random Uniform (5 docs/box, ~630 total) | 30 | Mean ± 95% CI reported |
| Skw | Random Skewed (NARA distribution) | 30 | Mean ± 95% CI reported |
| NoM | No Mask (all 31,681 docs) | 1 | Upper bound only |

### Evaluation
- **Metric:** Folder nDCG@5, averaged over 45 topics
- **Reporting:** Mean ± 95% CI
- **Significance:** Wilcoxon signed-rank test, paired by topic, p < 0.05. Mark significant results with *
- **Baseline to beat:** TOFS + BCE + AllF + SimSNC n=2 → **0.2141 ± 0.049 (Uni)** | **0.1563 ± 0.030 (Skw)**

### Folder Index Configurations (Phase 1)
| Code | Content indexed per folder |
|------|---------------------------|
| F1 | Folder Label only |
| F2 | Folder Label + Parent Expanded SNC |
| F3 | Folder Label + Scope Note (blank for folders without scope note) |
| F4 | Folder Label + Parent Expanded SNC + Scope Note |

### Query Types
| Code | Description | Use with |
|------|-------------|----------|
| Q0 | Normal TD (Title + Description) | All models |
| Q1 | HyDE — LLM generates SNC-style folder label | All models |
| Q2 | Keywords — entities, events, institutions list | **B, BC, BCE only** (never E alone) |
| Q3 | Embedding Text — descriptive historical paragraph | E, CE, BE, BCE |
| Q4 | Reformulations — 4 queries concatenated with original | All models |
| Q12 | Q1 + Q2 concatenated | B, BC, BCE |
| Q13 | Q1 + Q3 concatenated | E, CE, BE, BCE |
| Q14 | Q1 + Q4 concatenated | All models |
| Q-TCDE | Phase 1 query × 3 + 3 LLM topic passages | BCE (Phase 2C only) |

---

## Phase 0: Data Analysis

### Why
The Homophily augmentation strategy (Phase 2B) requires knowing which SNC codes have usable neighbor documents. Some SNCs may be "digitization deserts" where no training docs are available, making homophily infeasible. Phase 0 prevents wasted effort in Phase 2B and sets realistic expectations.

### ☐ Checklist

**0.1 SNC Distribution**
- [ ] Load `folder_metadata.json` for all 1,336 folders
- [ ] Count folders per 3-level SNC code (e.g., "POL 18"), per 2-level (e.g., "POL"), per 1-level primary code
- [ ] Plot histogram of folders per SNC
- [ ] List top-10 and bottom-10 SNCs by folder count
- [ ] Note: 379 distinct SNCs, 50 have scope notes — flag which SNCs benefit from F3/F4
- [ ] Distribution of documents to SNC (see if there are more favored SNC)

**0.2 Digitization Coverage per SNC**
- [ ] Load one representative Uniform ECF
- [ ] For each SNC, compute: (folders with ≥1 training doc) / (total folders with that SNC)
- [ ] Classify each SNC:
  - **Rich:** ≥50% of folders have training docs
  - **Moderate:** 20–49% have training docs
  - **Poor:** <20% have training docs
- [ ] Identify SNCs where >80% of folders are empty in Uniform ECFs (homophily won't help)

**0.3 Homophily Feasibility Study**
- [ ] For each folder without training docs (representative Uni ECF), count:
  - Same-SNC (exact) folders that DO have training docs
  - Same-SNC (2-level prefix, e.g., "POL 23") folders with training docs
  - Same-Box folders with training docs
- [ ] Compute distribution: "# of same-SNC docs available for the average empty folder"
- [ ] Answer: What % of empty folders have ≥3 same-SNC docs? ≥1? (thresholds for viability)
- [ ] **Fill Decision D6 and D7** based on this analysis

**0.4 Scope Note Coverage**
- [ ] Count: how many of the 1,336 folders have non-empty scope notes?
- [ ] List which SNC primary codes have scope notes (to predict F3/F4 benefit)
- [ ] Note this affects Phase 1 Set 1A: F3 and F4 may be sparse for many folders

**0.5 Date Range Summary**
- [ ] Compute distribution of folder date ranges (year of start date)
- [ ] Flag folders with missing or anomalous dates (may need fallback handling in prompts)

**0.6 Textual SNC**
- [ ] Check the different SNC values in order to see which have null values.

**0.7 Documents**
- [ ] Analyze some documents Title, Summary and OCR inside the SNC folders to see if they make sense.

### Required Outputs from Phase 0
1. SNC coverage table: `SNC → folder_count → avg_training_docs_per_ECF → digitization_category`
2. Homophily viability number: `% of empty folders with ≥{D7} same-SNC docs`
3. Filled decisions D6, D7, D8, D9 (see Future Decisions Tracker)
4. Insights about the SNCs and how they work.
5. Insights about the documents and their relation to the SNC.

---

## Phase 1: Query Expansion

### Objective
Find the best combination of **(a) folder index configuration** and **(b) query representation** for the AllF (All-Folder-Label) retrieval pipeline. Findings carry directly into Phase 2.

**Important:** Phase 1 experiments index ALL 1,336 folders using their ORIGINAL metadata (no LLM augmentation). They do NOT use any training documents. Therefore, they are evaluated once on 45 topics without ECF variation.

---

### Why Each Query Type Works

**Q1 — HyDE (adapted)**
Standard HyDE generates a hypothetical answer document. Here, we generate a hypothetical *SNC folder label* — a query in the same style as the index. Since the AllF index contains folder labels, a stylistically similar query increases exact-match probability for BM25F and creates a more calibrated embedding for dense models.

**Q2 — BM25 Keywords**
BM25F is a bag-of-words model. Short TD queries (avg. 13 words for Description) cover limited vocabulary. Generating a list of specific entities, people, and events expands term coverage dramatically. The folder labels are also short — term overlap is the critical signal for BM25F. *Do not use with E.*

**Q3 — Embedding Descriptive Text**
Embedding models compress the query into one vector. A rich 150-word descriptive paragraph gives the model more semantic content to work with and produces a more precise query vector than a short 13-word description. *Not useful for BM25F (length matters less than terms for sparse models).*

**Q4 — Reformulations (concatenated)**
Four reformulations from different angles increase the chance of matching the specific phrasing used in folder labels. Concatenation gives more tokens to BM25F (more term hits possible) and shifts the embedding centroid to cover multiple perspectives of the query.

---

### Code to Implement

```
src/query_expansion/
├── generate_hyde.py           # LLM → Q1 for all 45 topics → save queries_hyde.json
├── generate_keywords.py       # LLM → Q2 for all 45 topics → save queries_keywords.json
├── generate_emb_text.py       # LLM → Q3 for all 45 topics → save queries_embtext.json
├── generate_reformulations.py # LLM → Q4 for all 45 topics → save queries_reformulations.json
└── combine_queries.py         # Combines saved outputs → Q12, Q13, Q14, Q-TCDE

src/retrieval/
├── allf_index.py              # Builds index from folder metadata (F1–F4 configs)
├── allf_retrieval.py          # Runs retrieval: given query + index + model → ranked list
└── results_manager.py         # Saves ranked lists with metadata for later scoring
```

**LLM Caching:** Cache by `(topic_id, query_type)`. All 45 topics × 4 types = **180 LLM calls** total for Phase 1 query generation. Run once, store JSON, reuse forever.

---

### Prompts

---

#### PROMPT P1-HYDE
*Goal:* Generate a synthetic SNC-style folder label that represents the ideal folder for the query. This matches the style of the AllF index.

```
You are an expert on U.S. State Department records from the 1960s and 1970s, specifically diplomatic archives on Brazil. The State Department organized records using Subject-Numeric Codes (SNC). Here are representative examples of SNC folder labels from the collection:

- POL 15-1 BRAZ 01/01/1964: Executive branch of the Brazilian government
- POL 23-8 BRAZ 01/01/1965: Riots and civil unrest in Brazil
- DEF 19-3 BRAZ 01/01/1967: Military equipment and arms transfers to Brazil
- ECON 6 BRAZ 01/01/1963: Economic and financial statistics for Brazil
- LAB 3 BRAZ 01/01/1966: Labor conditions and union activities in Brazil
- AID 9 BRAZ 01/01/1964: U.S. economic and military assistance to Brazil
- POL 7 BRAZ 01/01/1969: Visits and meetings between government officials
- MIL 6 BRAZ 01/01/1968: Military exercises and operations in Brazil

Given the following research topic, generate 1 to 3 SNC-style folder labels that would describe the folders most likely to contain relevant documents.

Research Topic Title: {title}
Research Topic Description: {description}

Return ONLY the SNC-style labels, one per line. No explanations, no numbering, no preamble.
```

*Post-processing:* Concatenate all generated labels (space-separated) as the Q1 string.

---

#### PROMPT P1-KW
*Goal:* Generate a term-rich keyword list to maximize BM25F recall.

```
You are a research assistant specializing in U.S.-Brazil diplomatic history from the 1960s and 1970s. Generate a comprehensive list of search terms for retrieving relevant archival documents on the topic below.

Research Topic Title: {title}
Research Topic Description: {description}

Include in your list:
- Full names of specific people (politicians, diplomats, military officers, government officials relevant to this topic)
- Names of organizations and institutions (Brazilian government bodies, U.S. agencies, international bodies)
- Geographic locations specifically relevant to this topic
- Names of specific operations, programs, legislation, events, or incidents
- Technical terms and domain-specific jargon
- Common abbreviations used in State Department documents
- Key Portuguese-language names or terms that would appear in files

Return ONLY a single comma-separated list. No bullet points, no numbering, no explanations. Maximum 30 terms.
```

*Post-processing:* Replace commas with spaces for the Q2 query string.

---

#### PROMPT P1-EMB
*Goal:* Generate a rich descriptive paragraph to maximize semantic precision for dense retrieval.

```
You are a diplomatic historian specializing in U.S.-Brazil relations during the 1960s and 1970s. Write a detailed descriptive paragraph for the research topic below, as if writing for an academic encyclopedia on U.S. diplomatic archives.

Research Topic Title: {title}
Research Topic Description: {description}

Write a single paragraph of 150-200 words that:
- Describes the topic with historical specificity for the Brazil-U.S. context
- Mentions key actors, institutions, and events relevant to this topic during the 1960s-1970s
- Uses the terminology and phrasing characteristic of State Department documents
- Captures the political and diplomatic context a researcher would need
- Focuses on what kinds of archival records would satisfy this information need

Write only the paragraph. No title, no header, no preamble.
```

*Post-processing:* Use the paragraph directly as the Q3 string.

---

#### PROMPT P1-REF
*Goal:* Generate 4 complementary reformulations to capture different facets of the query.

```
You are an expert in archival research on U.S. State Department records on Brazil (1960s-1970s). Generate exactly 4 different search queries for the research topic below, each approaching it from a different angle.

Research Topic Title: {title}
Research Topic Description: {description}

Query 1: Focus on the official U.S. government perspective, actions, and policy
Query 2: Focus on Brazilian actors, institutions, and domestic context
Query 3: Focus on the specific events, operations, incidents, or policies involved
Query 4: Focus on key named individuals and organizations by their specific names

Rules:
- Each query must be 5-15 words long
- Use specific names and terms where possible
- Minimize word repetition across the 4 queries
- Return ONLY the 4 queries, one per line, without numbering or labels
```

*Post-processing:* Final Q4 string = `{title} {description} {title} {description} {query1} {query2} {query3} {query4}`
(TD repeated twice to preserve query weight before the reformulations.)

---

### Experiment Set 1A: Normal Queries × Folder Index Fields

**Purpose:** Identify the best folder metadata configuration for AllF retrieval.
**Query:** Q0 (TD — Title + Description, same as baseline)
**Evaluation:** nDCG@5 over 45 topics, single run

| Run ID | Index Config | Model | Compare Against |
|--------|-------------|-------|----------------|
| 1A-01 | F1 (Label) | B | — |
| 1A-02 | F1 | E | — |
| 1A-03 | F1 | C | — |
| 1A-04 | F1 | BCE | — |
| 1A-05 | F1 | BC | — |
| 1A-06 | F1 | BE | — |
| 1A-07 | F1 | CE | — |
| 1A-08 | F2 (Label + ParentSNC) | B | 1A-01 |
| 1A-09 | F2 | E | 1A-02 |
| 1A-10 | F2 | C | 1A-03 |
| 1A-11 | F2 | BCE | 1A-04 |
| 1A-12 | F2 | BC | 1A-05 |
| 1A-13 | F2 | BE | 1A-06 |
| 1A-14 | F2 | CE | 1A-07 |
| 1A-15 | F3 (Label + ScopeNote) | B | 1A-01 |
| 1A-16 | F3 | E | 1A-02 |
| 1A-17 | F3 | C | 1A-03 |
| 1A-18 | F3 | BCE | 1A-04 |
| 1A-19 | F3 | BC | 1A-05 |
| 1A-20 | F3 | BE | 1A-06 |
| 1A-21 | F3 | CE | 1A-07 |
| 1A-22 | F4 (Label + ParentSNC + ScopeNote) | B | 1A-01 |
| 1A-23 | F4 | E | 1A-02 |
| 1A-24 | F4 | C | 1A-03 |
| 1A-25 | F4 | BCE | 1A-04 |
| 1A-26 | F4 | BC | 1A-05 |
| 1A-27 | F4 | BE | 1A-06 |
| 1A-28 | F4 | CE | 1A-07 |

**Total 1A: 28 runs**

☐ **After 1A:** Record F_best for B, E, C, BCE individually. Note: F3/F4 effects are limited to 50/379 SNCs that have scope notes — may not show strong gains.

---

### Experiment Set 1B: Expanded Queries

**Purpose:** Identify the best query type for each model on AllF.
**Index:** F_best (best configuration per model from Set 1A; use F_best_BCE as default if uncertain)
**Evaluation:** nDCG@5 over 45 topics, single run

#### 1B-1: HyDE (Q1) — all models

| Run ID | Query | Model | Compare Against |
|--------|-------|-------|----------------|
| 1B-01 | Q1 | B | 1A best B result |
| 1B-02 | Q1 | E | 1A best E result |
| 1B-03 | Q1 | C | 1A best C result |
| 1B-04 | Q1 | BCE | 1A best BCE result |
| 1B-05 | Q1 | BC | 1A best BC result |
| 1B-06 | Q1 | BE | 1A best BE result |
| 1B-07 | Q1 | CE | 1A best CE result |

#### 1B-2: BM25 Keywords (Q2) — B, BC, BCE only

| Run ID | Query | Model | Compare Against |
|--------|-------|-------|----------------|
| 1B-08 | Q2 | B | 1A best B / 1B-01 |
| 1B-09 | Q2 | BC | 1A best BC / 1B-05 |
| 1B-10 | Q2 | BCE | 1A best BCE / 1B-04 |

#### 1B-3: Embedding Text (Q3) — E, CE, BE, BCE

| Run ID | Query | Model | Compare Against |
|--------|-------|-------|----------------|
| 1B-11 | Q3 | E | 1A best E / 1B-02 |
| 1B-12 | Q3 | CE | 1A best CE / 1B-07 |
| 1B-13 | Q3 | BE | 1A best BE / 1B-06 |
| 1B-14 | Q3 | BCE | 1A best BCE / 1B-04 |

#### 1B-4: Reformulations (Q4) — all models

| Run ID | Query | Model | Compare Against |
|--------|-------|-------|----------------|
| 1B-15 | Q4 | B | 1A best B / 1B-01 |
| 1B-16 | Q4 | E | 1A best E / 1B-02 |
| 1B-17 | Q4 | C | 1A best C / 1B-03 |
| 1B-18 | Q4 | BCE | 1A best BCE / 1B-04 |
| 1B-19 | Q4 | BC | 1A best BC / 1B-05 |
| 1B-20 | Q4 | BE | 1A best BE / 1B-06 |
| 1B-21 | Q4 | CE | 1A best CE / 1B-07 |

#### 1B-5: Query Combinations

| Run ID | Query | Model | Compare Against |
|--------|-------|-------|----------------|
| 1B-22 | Q12 (Q1+Q2) | B | Best of 1B-01, 1B-08 |
| 1B-23 | Q12 | BC | Best of 1B-05, 1B-09 |
| 1B-24 | Q12 | BCE | Best of 1B-04, 1B-10 |
| 1B-25 | Q13 (Q1+Q3) | E | Best of 1B-02, 1B-11 |
| 1B-26 | Q13 | CE | Best of 1B-07, 1B-12 |
| 1B-27 | Q13 | BE | Best of 1B-06, 1B-13 |
| 1B-28 | Q13 | BCE | Best of 1B-04, 1B-14 |
| 1B-29 | Q14 (Q1+Q4) | B | Best of 1B-01, 1B-15 |
| 1B-30 | Q14 | E | Best of 1B-02, 1B-16 |
| 1B-31 | Q14 | C | Best of 1B-03, 1B-17 |
| 1B-32 | Q14 | BCE | Best of 1B-04, 1B-18 |
| 1B-33 | QALL (Q1+Q2+Q3+Q4) | BCE | Best BCE so far |

**Total 1B: 33 runs | Phase 1 Total: 61 runs**

### Phase 1 Decisions (fill after running)
| Decision | Value |
|---------|-------|
| D1 — F_best (overall) | ___ |
| D1a — F_best for B | ___ |
| D1b — F_best for E | ___ |
| D1c — F_best for C | ___ |
| D2 — Q_best_B | ___ |
| D3 — Q_best_E | ___ |
| D4 — Q_best_C | ___ |
| D5 — Q_best_BCE | ___ |

> These five values are carried into ALL Phase 2 experiments. The learning from Phase 1 reduces the query search space in Phase 2.

---

## Phase 2: Folder Label Augmentation

### Overview
Phase 2 replaces the original folder labels with LLM-generated augmented labels. The AllF pipeline then searches these enriched representations instead of the original labels.

**Key difference from Phase 1:** Augmented labels for folders *with* training documents depend on which documents are in the ECF. Therefore, all Phase 2 experiments are evaluated over 30 Uniform ECFs and 30 Skewed ECFs (mean ± 95% CI per mask).

### ECF-Dependency and Caching Logic

For any given ECF:
- Folder has training docs → use document-aware augmentation prompt (content varies per ECF)
- Folder has NO training docs → use pure metadata/knowledge prompt (content is ECF-independent)

**Optimization:** Cache by `(folder_id, frozenset(training_doc_ids))`. The same document set appearing in multiple ECFs for the same folder reuses the cached output. Folders without docs are generated once and shared across all ECFs.

**Pilot strategy:** Run the full augmentation pipeline on 5 ECFs first. Review outputs before committing to all 30.

### Augmented Label Index Format
After generation, each folder's entry in the retrieval index will be:
```
{original_folder_label} | {expanded_snc} | {DESCRIPTION} | {KEYWORDS}
```
The original label is preserved as a prefix (exact-match fallback) while the enriched content follows.

---

### Strategy 2A: Base Augmentation (Qualification Methodology)

#### Why
The LLM acts as an informed archivist. For folders with digitized documents, it grounds the augmented label in actual OCR evidence — extracting specific people, events, and dates. For empty folders, it uses the SNC classification and its historical knowledge of the 1960s-1970s Brazil-U.S. context. This directly bridges the semantic gap between brief SNC codes ("POL 18 Pernambuco 1964") and natural-language user queries ("political events in Pernambuco"). The key insight from the guiding principle: *ask the LLM to enumerate specific events first, then synthesize — never ask for enrichment in a single open-ended step.*

#### Document Content to Use
Use `title + summary` per document (GPT-4o summaries already exist in the collection). If a folder has >5 training documents, use the top-5 by BM25 score against the folder label (as a quick proxy for most representative). Fill decision D13 after reviewing pilot outputs.

#### Code to Implement

```
src/augmentation/aug_base.py
  ├── load_ecf(ecf_path)
  │     → {folder_id: [doc_ids]}  (folders that have docs in this ECF)
  ├── get_doc_content(doc_id, fields=['title','summary'])
  │     → str  (concatenated fields)
  ├── select_docs_for_folder(folder_id, doc_ids, max_docs=5)
  │     → [doc_id, ...]  (top-5 by BM25 relevance to folder label)
  ├── augment_with_docs(folder_meta, doc_texts, llm_client)
  │     → str  (DESCRIPTION + KEYWORDS)
  ├── augment_no_docs(folder_meta, llm_client)
  │     → str  (DESCRIPTION + KEYWORDS)
  └── run_augmentation(ecf_path, llm_client, cache_path)
        → {folder_id: augmented_label_str}  (all 1,336 folders)
```

---

#### PROMPT P2A-DOC (Folders WITH Training Documents)

```
You are an expert archivist and historian specializing in U.S.-Brazil diplomatic relations during the 1960s and 1970s. Write enriched metadata for the archival folder below.

FOLDER INFORMATION:
- Folder Label: {folder_label}
- SNC Code: {snc}
- SNC Meaning: {expanded_snc}
- Broader Category: {parent_expanded_snc}
- Scope Note: {scope_note}
- Date Range: {start_date} to {end_date}
- Record Group: {record_group}

DOCUMENTS IN THIS FOLDER:
{doc_list}
[Format: [Doc N] Title: ... | Summary: ...]

Think through these steps before writing:
Step 1 — ENTITIES: List the specific people, organizations, institutions, and locations mentioned across the documents.
Step 2 — EVENTS: List the specific events, actions, decisions, or developments described.
Step 3 — THEMES: Identify 2-3 overarching themes that connect the documents.

Then output the following two sections:

DESCRIPTION: A paragraph of 100-150 words starting exactly with "This folder contains" describing the specific historical content, key actors, events, and context in these documents. Be specific — name individuals, institutions, and events. Do not use generic phrases like "various documents" or "multiple topics."

KEYWORDS: A comma-separated list of 10-15 search terms: proper nouns, event names, institutions, and key concepts that best represent this folder's content.

Output only the DESCRIPTION and KEYWORDS sections. Do not output your step-by-step reasoning.
```

---

#### PROMPT P2A-NODOC (Folders WITHOUT Training Documents)

```
You are an expert archivist and historian specializing in U.S.-Brazil diplomatic relations during the 1960s and 1970s. Write enriched metadata for an archival folder that has not yet been digitized.

FOLDER INFORMATION:
- Folder Label: {folder_label}
- SNC Code: {snc}
- SNC Meaning: {expanded_snc}
- Broader Category: {parent_expanded_snc}
- Scope Note: {scope_note}
- Date Range: {start_date} to {end_date}
- Record Group: {record_group}

HISTORICAL CONTEXT:
This folder is from U.S. State Department records on Brazil. The period {start_date}–{end_date} includes: the April 1964 military coup and subsequent military government, Cold War dynamics in Latin America, significant Brazilian economic development, and evolving U.S.-Brazil bilateral relations across security, trade, and diplomacy.

Think through these steps before writing:
Step 1 — SNC INTERPRETATION: What specific type of content does this SNC code cover? What would a State Department officer file under this label?
Step 2 — HISTORICAL EVENTS: What specific events, actors, and developments occurred in Brazil and in U.S.-Brazil relations relevant to this SNC topic during {start_date}–{end_date}?
Step 3 — DOCUMENT INFERENCE: What types of documents (cables, memos, intelligence reports, diplomatic notes) would likely populate this folder?

Then output the following two sections:

DESCRIPTION: A paragraph of 100-150 words starting exactly with "This folder likely contains" describing the probable historical content, likely key actors and institutions, and relevant events. Name known historical figures, institutions, and events from this era. Do not invent document contents — draw only from established historical knowledge.

KEYWORDS: A comma-separated list of 10-15 search terms including proper nouns, event names, institutions, and key concepts most likely to represent this folder.

Output only the DESCRIPTION and KEYWORDS sections. Do not output your step-by-step reasoning.
```

---

#### Experiment Set 2A

*Each run is averaged over 30 ECFs per mask. In the run table: Uni = 30 Uniform ECFs; Skw = 30 Skewed ECFs.*

| Run ID | Augmentation | Query | Model | Mask | Compare Against |
|--------|-------------|-------|-------|------|----------------|
| 2A-01 | Aug-Base | Q0 | B | Uni | 1A best B (AllF baseline) |
| 2A-02 | Aug-Base | Q0 | E | Uni | 1A best E |
| 2A-03 | Aug-Base | Q0 | C | Uni | 1A best C |
| 2A-04 | Aug-Base | Q0 | BCE | Uni | 1A best BCE |
| 2A-05 | Aug-Base | Q0 | BC | Uni | 1A best BC |
| 2A-06 | Aug-Base | Q0 | BE | Uni | 1A best BE |
| 2A-07 | Aug-Base | Q0 | CE | Uni | 1A best CE |
| 2A-08 | Aug-Base | Q_best_B | B | Uni | 2A-01 |
| 2A-09 | Aug-Base | Q_best_E | E | Uni | 2A-02 |
| 2A-10 | Aug-Base | Q_best_C | C | Uni | 2A-03 |
| 2A-11 | Aug-Base | Q_best_BCE | BCE | Uni | 2A-04 |
| 2A-12 | Aug-Base | Q_best_BCE | BC | Uni | 2A-05 |
| 2A-13 | Aug-Base | Q_best_BCE | BE | Uni | 2A-06 |
| 2A-14 | Aug-Base | Q_best_BCE | CE | Uni | 2A-07 |
| 2A-15 | Aug-Base | Q0 | B | Skw | 2A-01 (cross-mask) |
| 2A-16 | Aug-Base | Q0 | E | Skw | 2A-02 |
| 2A-17 | Aug-Base | Q0 | C | Skw | 2A-03 |
| 2A-18 | Aug-Base | Q0 | BCE | Skw | 2A-04 |
| 2A-19 | Aug-Base | Q_best_B | B | Skw | 2A-15 |
| 2A-20 | Aug-Base | Q_best_E | E | Skw | 2A-16 |
| 2A-21 | Aug-Base | Q_best_C | C | Skw | 2A-17 |
| 2A-22 | Aug-Base | Q_best_BCE | BCE | Skw | 2A-18 |

**Total 2A: 22 runs (× 30 ECFs per mask = 1,320 retrieval executions)**

---

### Strategy 2B: Homophily Augmentation

#### Why
The archival principle of *original order* implies that folders filed under the same SNC code share thematic content. When an empty folder shares an SNC code with folders that DO have training documents, those documents serve as thematic proxies. Including neighbor-folder document evidence in the prompt gives the LLM collection-specific vocabulary and events — more grounded than relying purely on general historical knowledge. This specifically targets the weakness of 2A for empty folders: generic text from knowledge alone vs. collection-specific text from similar folders.

#### Prerequisite
**Requires completed Phase 0, Sections 0.2 and 0.3.** Fill decisions D6, D7, D8 before implementing.

#### Decisions to Fill (from Phase 0)
- D6 — Similarity criterion: ☐ Same SNC (exact) / ☐ Same SNC 2-level prefix / ☐ Same Box / ☐ Both: ___
- D7 — Minimum neighbor docs threshold: ___ (suggested: 3)
- D8 — Max neighbor docs in prompt: ___ (suggested: 5)
- D9 — Proceed with Homophily? ☐ Yes / ☐ No (if <20% of empty folders pass threshold) / ☐ Hybrid: ___

**Fallback:** For empty folders that don't meet the neighbor threshold → use P2A-NODOC.

#### Code to Implement

```
src/augmentation/aug_homophily.py
  ├── find_neighbor_folders(folder_id, snc, box_id, all_folders_with_docs, strategy=D6)
  │     → [(neighbor_folder_id, doc_ids), ...]
  ├── select_neighbor_docs(neighbor_list, max_docs=D8)
  │     → [doc_id, ...]
  ├── augment_with_neighbors(folder_meta, neighbor_doc_texts, llm_client)
  │     → str  (DESCRIPTION + KEYWORDS)
  └── run_homophily_augmentation(ecf_path, llm_client, cache_path, threshold=D7)
        → {folder_id: augmented_label_str}
        # Folders with own docs → same as P2A-DOC
        # Folders without docs but ≥ threshold neighbors → P2B-HOMO
        # Folders without docs and < threshold neighbors → P2A-NODOC
```

---

#### PROMPT P2B-HOMO (Folders WITHOUT Documents, with Neighbor Context)

```
You are an expert archivist and historian specializing in U.S.-Brazil diplomatic relations during the 1960s and 1970s. Write enriched metadata for an archival folder that has not been digitized.

TARGET FOLDER (not yet digitized):
- Folder Label: {folder_label}
- SNC Code: {snc}
- SNC Meaning: {expanded_snc}
- Date Range: {start_date} to {end_date}

CONTEXTUAL DOCUMENTS FROM RELATED FOLDERS:
The documents below come from OTHER folders with the same SNC classification ({snc}). They do NOT belong to the target folder. Use them to understand what topics, actors, and events are typically filed under this code — but do not assume their specific events apply to the target folder.

{neighbor_docs}
[Format: [Related Doc N] From Folder: {folder_label} | Title: ... | Summary: ...]

Think through these steps before writing:
Step 1 — SNC PATTERNS: Based on the related documents, what recurring topics, actors, and events appear under this SNC code?
Step 2 — TARGET PERIOD SPECIFICITY: What events specifically occurred in {start_date}–{end_date} that align with this SNC topic? Use historical knowledge to situate the target folder in time.
Step 3 — LABEL DISTINCTION: The target folder is specifically labeled "{folder_label}". What distinguishes it from the related folders? Focus on the geographic or topical qualifier in the label.

Then output the following two sections:

DESCRIPTION: A paragraph of 100-150 words starting exactly with "This folder likely contains" combining insights from the related documents with the target folder's specific label and time period. Reference specific historical events and actors relevant to this exact label and period. Do not copy events directly from the related documents — use them as thematic context only.

KEYWORDS: A comma-separated list of 10-15 search terms derived from the target folder's SNC topic and the patterns observed in related documents.

Output only the DESCRIPTION and KEYWORDS sections. Do not output your step-by-step reasoning.
```

---

#### Experiment Set 2B

*For folders WITH own docs: P2A-DOC is used (same as 2A). Homophily only changes the empty-folder augmentation.*

| Run ID | Augmentation | Query | Model | Mask | Compare Against |
|--------|-------------|-------|-------|------|----------------|
| 2B-01 | Aug-Homo | Q0 | BCE | Uni | 2A-04 (same query, different empty-folder aug) |
| 2B-02 | Aug-Homo | Q_best_BCE | BCE | Uni | 2B-01 |
| 2B-03 | Aug-Homo | Q0 | B | Uni | 2A-01 |
| 2B-04 | Aug-Homo | Q_best_B | B | Uni | 2B-03 |
| 2B-05 | Aug-Homo | Q0 | E | Uni | 2A-02 |
| 2B-06 | Aug-Homo | Q_best_E | E | Uni | 2B-05 |
| 2B-07 | Aug-Homo | Q0 | C | Uni | 2A-03 |
| 2B-08 | Aug-Homo | Q_best_C | C | Uni | 2B-07 |
| 2B-09 | Aug-Homo | Q0 | BCE | Skw | 2A-18 |
| 2B-10 | Aug-Homo | Q_best_BCE | BCE | Skw | 2B-09 |
| 2B-11 | Aug-Homo | Q0 | B | Skw | 2A-15 |
| 2B-12 | Aug-Homo | Q_best_B | B | Skw | 2B-11 |

**Total 2B: 12 runs** — *Cancel this set entirely if D9 = No (viability <20%)*

---

### Strategy 2C: TCDE Augmentation

#### Why
TCDE (Topic-Centric Dual Expansion) enforces structural specificity via a two-pass LLM process. The first pass extracts exactly 5 distinct abstract topics from the folder. The second pass generates one detailed, entity-rich expansion sentence per topic. This prevents the "generic paragraph" problem because each topic sentence anchors the second pass to a specific aspect of the folder's content. The result is a multi-faceted representation covering different dimensions, which improves both sparse retrieval (more term diversity) and dense retrieval (better semantic coverage). Applied to both the folder (document side) and the query (query side), TCDE creates matched vocabulary between query and index.

#### Two-Pass Flow
```
Pass 1 (P2C-TOPIC): folder metadata + docs (if any) → 5 topic sentences
Pass 2 (P2C-EXPAND): 5 topic sentences + folder context → 5 expansion sentences
Final TCDE label: all 5 topic sentences + all 5 expansion sentences concatenated
```

#### Optional: TCDE Query Expansion (Q-TCDE)
A matching query-side expansion to pair with TCDE folder labels:
```
Original TD (repeated ×3) + 3 LLM-generated topic passages for the query
```
This is tested only in Set 2C experiments (runs 2C-03 and 2C-12).

#### Code to Implement

```
src/augmentation/aug_tcde.py
  ├── identify_topics(folder_meta, doc_texts_or_none, llm_client)
  │     → [topic_1, topic_2, topic_3, topic_4, topic_5]
  ├── expand_topics(topic_list, folder_meta, llm_client)
  │     → [expansion_1, ..., expansion_5]
  ├── generate_tcde_label(folder_meta, doc_texts, llm_client)
  │     → str  (topics concatenated + expansions concatenated)
  └── run_tcde_augmentation(ecf_path, llm_client, cache_path)
        → {folder_id: augmented_label_str}

src/query_expansion/generate_tcde_query.py
  ├── generate_tcde_passages(query_td, llm_client, n=3)
  │     → [passage_1, passage_2, passage_3]
  └── build_tcde_query(title, description, passages)
        → str  (TD×3 + passages concatenated)
```

---

#### PROMPT P2C-TOPIC (Pass 1: Topic Identification)

```
You are an expert archivist specializing in U.S. State Department records on Brazil (1960s-1970s). Identify exactly 5 abstract topics for the archival folder below.

FOLDER INFORMATION:
- Label: {folder_label}
- SNC: {expanded_snc}
- Date Range: {start_date} to {end_date}
{[IF HAS DOCS]}
DOCUMENT CONTENT (sample):
{doc_texts}
{[ELSE]}
No documents are digitized for this folder. Base your analysis on the folder label, SNC classification, and knowledge of the {start_date}–{end_date} period.
{[END IF]}

Identify exactly 5 DISTINCT topics. Each topic must:
- Be expressed as one specific sentence (not a general category)
- Cover a different aspect of the folder's likely content
- Reference specific types of actors, events, or issues (not generic phrases)
- Be grounded in the {start_date}–{end_date} historical context

Return exactly 5 topic sentences. One per line. No numbering, no preamble.
```

---

#### PROMPT P2C-EXPAND (Pass 2: Topic Expansion)

```
You are a diplomatic historian specializing in U.S.-Brazil relations (1960s-1970s). Below are 5 topics for an archival folder ({folder_label}, {start_date}–{end_date}).

TOPICS:
{topic_1}
{topic_2}
{topic_3}
{topic_4}
{topic_5}

For each topic, write one detailed expansion sentence that:
- Names specific historical actors, institutions, or events related to that topic
- Uses terminology consistent with U.S. State Department language from the 1960s-1970s
- Provides concrete historical context useful for a researcher searching these archives

Return exactly 5 sentences, one per line, in the same order as the topics. No numbering, no preamble.
```

---

#### PROMPT P2C-QEXP (TCDE Query-Side Expansion — for Q-TCDE)

```
You are a research assistant specializing in U.S.-Brazil diplomatic archives (1960s-1970s). Given the research topic below, generate 3 diverse passages that each explore a different dimension of the same information need. Each passage should be 2-3 sentences and simulate the kind of description that might appear in an archival finding aid.

Research Topic Title: {title}
Research Topic Description: {description}

Generate exactly 3 passages, each from a different angle: (1) institutional/policy, (2) actors/people, (3) events/actions.

Return the 3 passages separated by a blank line. No numbering, no labels.
```

*Post-processing:* Q-TCDE = `{title} {description}` × 3 + all 3 passages concatenated

---

#### Experiment Set 2C

| Run ID | Augmentation | Query | Model | Mask | Compare Against |
|--------|-------------|-------|-------|------|----------------|
| 2C-01 | Aug-TCDE | Q0 | BCE | Uni | 2A-04 (base aug, same query) |
| 2C-02 | Aug-TCDE | Q_best_BCE | BCE | Uni | 2C-01 |
| 2C-03 | Aug-TCDE | Q-TCDE | BCE | Uni | 2C-02 (TCDE folder + TCDE query) |
| 2C-04 | Aug-TCDE | Q0 | B | Uni | 2A-01 |
| 2C-05 | Aug-TCDE | Q_best_B | B | Uni | 2C-04 |
| 2C-06 | Aug-TCDE | Q0 | E | Uni | 2A-02 |
| 2C-07 | Aug-TCDE | Q_best_E | E | Uni | 2C-06 |
| 2C-08 | Aug-TCDE | Q0 | C | Uni | 2A-03 |
| 2C-09 | Aug-TCDE | Q_best_C | C | Uni | 2C-08 |
| 2C-10 | Aug-TCDE | Q0 | BCE | Skw | 2A-18 |
| 2C-11 | Aug-TCDE | Q_best_BCE | BCE | Skw | 2C-10 |
| 2C-12 | Aug-TCDE | Q-TCDE | BCE | Skw | 2C-11 |

**Total 2C: 12 runs**

---

### Cross-Strategy Comparison

After 2A, 2B, 2C, test whether combining strategies for different folder tiers outperforms any single strategy.

| Run ID | Augmentation | Query | Model | Mask | Logic |
|--------|-------------|-------|-------|------|-------|
| 2X-01 | Hybrid (2A-doc + 2C-nodoc) | Q_best_BCE | BCE | Uni | Folders with docs → P2A-DOC; empty → P2C |
| 2X-02 | Hybrid (2A-doc + 2B-nodoc) | Q_best_BCE | BCE | Uni | Folders with docs → P2A-DOC; empty → P2B-HOMO |
| 2X-03 | Hybrid (2A-doc + 2C-nodoc) | Q_best_BCE | BCE | Skw | Same as 2X-01 for Skewed |

**Total 2X: 3 runs**

### Phase 2 Decisions (fill after running)

| Decision | Value |
|---------|-------|
| D10 — Aug_best (best single strategy, Uni) | ___ |
| D10a — Aug_best for Skewed | ___ |
| D11 — Does Hybrid outperform single strategy? | ☐ Yes / ☐ No |
| D12 — N topics for TCDE | ___ (5 recommended, adjust if pilot shows issues) |
| D13 — Document content for prompts | ☐ title+summary / ☐ title+OCR-p1 / ☐ full-OCR |
| D14 — LLM for augmentation | ___ |

---

## Phase 3: Re-ranking

### Objective
Re-rank the top-100 folders from the best Phase 2 run. Phase 2 gives an initial ranking based on augmented folder labels. Phase 3 adds a second signal: actual training document scores (3A) or LLM reasoning (3B).

### Why Re-ranking Helps
Phase 2 augmented labels are LLM-inferred — useful but imperfect. For folders that DO have training documents, scoring those documents directly against the query gives harder evidence of relevance. For folders without documents, LLM reasoning about which candidate is more likely to contain relevant material applies global context that the vector similarity alone cannot.

### Code to Implement

```
src/reranking/rerank_traditional.py
  ├── load_top_k(run_id, k=100) → [(folder_id, init_score), ...]
  ├── get_folder_docs(folder_id, ecf) → [(doc_id, doc_text), ...]
  ├── score_folder_by_docs(query, doc_texts, model) → float  (max doc score)
  └── run_traditional_rerank(query, top_k, ecf, model)
        → reranked [(folder_id, new_score), ...]
        # Scoring: if folder has docs → max doc score; if no docs → init_score × decay_factor

src/reranking/rerank_llm.py
  ├── load_top_k_with_context(run_id, k) → list of folder dicts (label, aug, docs)
  ├── build_rerank_prompt(query, folder_candidates) → str
  ├── parse_llm_ranking(llm_response) → [folder_id, ...]
  └── run_llm_rerank(query, folder_candidates, llm_client, k)
        → reranked [folder_id, ...]
```

---

### Strategy 3A: Traditional Document-Based Re-ranking

#### Why
This is the "document-aware second stage" from the qualification proposal, but evaluated at a larger cutoff (k=100). Folders that have training documents confirming relevance are boosted; folders without documents keep their Phase 2 score (with a small decay to discourage noise).

#### Scoring Rule
```
if folder has training docs:
    score = max(BM25F_score(query, doc) for doc in training_docs)
else:
    score = initial_phase2_score * decay_factor  (e.g., 0.9)
```

Then re-sort top-100 by new scores. Apply Reciprocal Rank Fusion if multiple models are used for document scoring.

#### Experiment Set 3A

| Run ID | Base Phase 2 Run | Re-rank Model | k | Mask | Compare Against |
|--------|-----------------|--------------|---|------|----------------|
| 3A-01 | Best 2A (Uni) | B | 100 | Uni | Best 2A run |
| 3A-02 | Best 2A (Uni) | E | 100 | Uni | 3A-01 |
| 3A-03 | Best 2A (Uni) | C | 100 | Uni | 3A-01 |
| 3A-04 | Best 2A (Uni) | BCE | 100 | Uni | 3A-01 |
| 3A-05 | Best 2B (Uni) | BCE | 100 | Uni | Best 2B run |
| 3A-06 | Best 2C (Uni) | BCE | 100 | Uni | Best 2C run |
| 3A-07 | Best 2A (Skw) | BCE | 100 | Skw | Best 2A Skw run |
| 3A-08 | Best 2C (Skw) | BCE | 100 | Skw | Best 2C Skw run |

**Total 3A: 8 runs**

---

### Strategy 3B: LLM-Based Re-ranking

#### Why
LLM re-ranking is *listwise*: all top-k candidates are presented together and the LLM reasons about relative relevance in context. The LLM can apply historical knowledge about which SNC topics are most likely to yield relevant materials for a given research query. It can also weight documentary evidence appropriately (a folder with confirmed documents > a folder with only a matching augmented label). This directly addresses the hard topics where standard retrieval fails.

---

#### PROMPT P3B-RERANK

```
You are an expert on U.S. State Department records and Brazilian diplomatic history (1960s-1970s). A researcher is searching for the most relevant archival folders. You have been given a ranked list of candidate folders. Re-rank them from most to least relevant.

RESEARCH TOPIC:
Title: {title}
Description: {description}

CANDIDATE FOLDERS (initial order — do not treat order as ground truth):
{folder_list}

[Format per folder:]
[#{rank}] ID: {folder_id}
  Label: {folder_label} | Classification: {expanded_snc} | Period: {date_range}
  Documents Available: {yes — N docs / no}
  Description: {augmented_description}
  Keywords: {augmented_keywords}

REASONING GUIDE:
Think through these questions:
1. Which folders have labels and descriptions that directly address the research topic?
2. Among those, do any folders have actual documentary evidence (docs available)? Those have confirmed relevance.
3. Among folders with matching labels but no documents: how confidently can you infer relevance from the SNC classification and historical context alone?
4. Are there any folders currently ranked low that should be promoted based on their historical fit with the topic?

Re-rank all {k} folders. Return ONLY the folder IDs in order, one per line:
1. {folder_id}
2. {folder_id}
...{k}. {folder_id}

After the ranked list, add one line:
REASONING: [One sentence explaining the primary factor driving your top-3 selections]
```

---

#### Experiment Set 3B

| Run ID | Base Phase 2 Run | LLM | k | Mask | Compare Against |
|--------|-----------------|-----|---|------|----------------|
| 3B-01 | Best 2A (Uni) | {D15} | 20 | Uni | 3A-04, Best 2A run |
| 3B-02 | Best 2A (Uni) | {D15} | 50 | Uni | 3B-01 |
| 3B-03 | Best 2A (Uni) | {D15} | 100 | Uni | 3B-02 |
| 3B-04 | Best 2C (Uni) | {D15} | 20 | Uni | 3A-06, Best 2C run |
| 3B-05 | Best 2C (Uni) | {D15} | 50 | Uni | 3B-04 |
| 3B-06 | Best 2A (Skw) | {D15} | 20 | Skw | 3A-07, Best 2A Skw |
| 3B-07 | Best 2C (Skw) | {D15} | 20 | Skw | 3A-08, Best 2C Skw |

**Total 3B: 7 runs** — Start with k=20 to control cost; scale up only if gains are confirmed.

### Phase 3 Decisions (fill after running)

| Decision | Value |
|---------|-------|
| D11 — Optimal k for re-ranking | ___ |
| D15 — LLM for re-ranking | ___ |
| D16 — Does 3B (LLM) outperform 3A (traditional) significantly? | ☐ Yes / ☐ No |
| D17 — Decay factor for no-doc folders in 3A | ___ (test 0.8, 0.9, 0.95) |

---

## Master Run Registry

| Run ID | Phase | Strategy | Query | Model | Mask | nDCG@5 | ±CI | vs Baseline* |
|--------|-------|---------|-------|-------|------|--------|-----|------------|
| **Baseline** | — | TOFS+BCE+AllF+SimSNC | Q0 | BCE | Uni | 0.2141 | 0.049 | — |
| **Baseline** | — | TOFS+BCE+AllF+SimSNC | Q0 | BCE | Skw | 0.1563 | 0.030 | — |
| 1A-01 | 1A | AllF-F1 | Q0 | B | — | | | |
| 1A-04 | 1A | AllF-F1 | Q0 | BCE | — | | | |
| 1A-11 | 1A | AllF-F2 | Q0 | BCE | — | | | |
| 1A-18 | 1A | AllF-F3 | Q0 | BCE | — | | | |
| 1A-25 | 1A | AllF-F4 | Q0 | BCE | — | | | |
| 1B-04 | 1B | AllF | Q1 | BCE | — | | | |
| 1B-10 | 1B | AllF | Q2 | BCE | — | | | |
| 1B-14 | 1B | AllF | Q3 | BCE | — | | | |
| 1B-18 | 1B | AllF | Q4 | BCE | — | | | |
| 1B-33 | 1B | AllF | QALL | BCE | — | | | |
| 2A-04 | 2A | Aug-Base | Q0 | BCE | Uni | | | |
| 2A-11 | 2A | Aug-Base | Q_best_BCE | BCE | Uni | | | |
| 2A-18 | 2A | Aug-Base | Q0 | BCE | Skw | | | |
| 2A-22 | 2A | Aug-Base | Q_best_BCE | BCE | Skw | | | |
| 2B-01 | 2B | Aug-Homo | Q0 | BCE | Uni | | | |
| 2B-02 | 2B | Aug-Homo | Q_best_BCE | BCE | Uni | | | |
| 2C-01 | 2C | Aug-TCDE | Q0 | BCE | Uni | | | |
| 2C-03 | 2C | Aug-TCDE | Q-TCDE | BCE | Uni | | | |
| 2X-01 | 2X | Hybrid | Q_best_BCE | BCE | Uni | | | |
| 3A-04 | 3A | Rerank-Trad | Q_best_BCE | BCE | Uni | | | |
| 3B-01 | 3B | Rerank-LLM k=20 | Q_best_BCE | BCE | Uni | | | |
| 3B-02 | 3B | Rerank-LLM k=50 | Q_best_BCE | BCE | Uni | | | |

*Fill with significance marker (* = p<0.05 vs chosen comparison) after running.

---

## Implementation Timeline

### Week 1-2: Infrastructure + Phase 0
- [ ] Implement `src/utils/llm_client.py` with disk-based caching (key by SHA256 of prompt)
- [ ] Implement `src/utils/results_manager.py` (save/load ranked lists keyed by run ID)
- [ ] Implement `src/utils/ecf_loader.py`
- [ ] Run Phase 0 data analysis scripts
- [ ] Fill decisions D6, D7, D8, D9

### Week 3: Phase 1 Query Generation
- [ ] Implement all 4 query generators; test on 3 topics before full run
- [ ] Run on all 45 topics → save `queries_hyde.json`, `queries_keywords.json`, `queries_embtext.json`, `queries_reformulations.json`
- [ ] Implement `combine_queries.py` → generate Q12, Q13, Q14, QALL for all 45 topics

### Week 3-4: Phase 1 Retrieval
- [ ] Implement `allf_index.py` (F1–F4 index builder)
- [ ] Implement `allf_retrieval.py` (supports B, E, C, BCE, BC, BE, CE)
- [ ] Run Set 1A (28 runs) → fill D1a, D1b, D1c, D1
- [ ] Run Set 1B (33 runs) → fill D2, D3, D4, D5

### Week 5-6: Phase 2A
- [ ] Implement `aug_base.py`; test P2A-DOC and P2A-NODOC on 5 folders (2 with docs, 3 without)
- [ ] Review LLM outputs for specificity — refine prompts if outputs are generic
- [ ] Run augmentation for 5 pilot Uniform ECFs → check quality
- [ ] Scale to all 30 Uniform ECFs + 30 Skewed ECFs (with caching)
- [ ] Run retrieval experiments Set 2A (22 runs)

### Week 7: Phase 2B (conditional on D9)
- [ ] If D9 = Yes: implement `aug_homophily.py`; test on 5 empty folders
- [ ] Run augmentation for all ECFs
- [ ] Run Set 2B (12 runs)

### Week 8: Phase 2C
- [ ] Implement `aug_tcde.py` (two-pass)
- [ ] Test P2C-TOPIC + P2C-EXPAND on 5 folders → verify 5 distinct topics per folder
- [ ] Generate Q-TCDE queries for all 45 topics
- [ ] Run augmentation for all ECFs
- [ ] Run Set 2C (12 runs) + 2X cross-strategy (3 runs)

### Week 9: Phase 3
- [ ] Implement `rerank_traditional.py`; run Set 3A (8 runs)
- [ ] Implement `rerank_llm.py` with P3B-RERANK; run Set 3B starting with k=20 (7 runs)
- [ ] Fill D11, D15, D16, D17

### Week 10: Analysis + Writing
- [ ] Run Wilcoxon signed-rank tests for all key pairwise comparisons
- [ ] Build comparison table: Phase 1 best → Phase 2 best → Phase 3 best → Baseline
- [ ] Analyze per-topic: identify which formerly hard topics are now resolved by augmentation or re-ranking
- [ ] Analyze by folder class: with-docs vs without-docs nDCG@5 (compute separately)
- [ ] Write dissertation sections

---

## Future Decisions Tracker

| # | Decision | Needed Before | Options | Fill-In |
|---|---------|--------------|---------|---------|
| D1 | F_best (best folder index config, overall) | Phase 1B | F1, F2, F3, F4 | ___ |
| D1a | F_best for B | Phase 1B | — | ___ |
| D1b | F_best for E | Phase 1B | — | ___ |
| D1c | F_best for C | Phase 1B | — | ___ |
| D2 | Q_best_B | Phase 2A | Q0, Q1, Q2, Q4, Q12, Q14 | ___ |
| D3 | Q_best_E | Phase 2A | Q0, Q1, Q3, Q13 | ___ |
| D4 | Q_best_C | Phase 2A | Q0, Q1, Q4, Q14 | ___ |
| D5 | Q_best_BCE | Phase 2A | Any combination | ___ |
| D6 | Homophily similarity criterion | Phase 2B | same-SNC exact / 2-level prefix / same-box / both | ___ |
| D7 | Homophily min neighbor docs threshold | Phase 2B | 1 / 2 / 3 / 5 | ___ |
| D8 | Max neighbor docs in P2B-HOMO prompt | Phase 2B | 3 / 5 / 10 | ___ |
| D9 | Proceed with Homophily strategy? | Phase 2B | Yes / No / Hybrid fallback | ___ |
| D10 | Aug_best (best augmentation, Uniform) | Phase 3 | 2A / 2B / 2C / Hybrid | ___ |
| D10a | Aug_best (best augmentation, Skewed) | Phase 3 | 2A / 2B / 2C / Hybrid | ___ |
| D11 | Optimal k for re-ranking | Phase 3 (after 3B k=20) | 20 / 50 / 100 | ___ |
| D12 | N topics per folder for TCDE | Week 8 pilot | 3 / 5 / 7 | ___ |
| D13 | Document content for augmentation | Week 5 pilot | title+summary / title+OCR-p1 / full-OCR | ___ |
| D14 | LLM model for augmentation | Week 1 | GPT-4o / Claude Sonnet / Claude Opus | ___ |
| D15 | LLM model for re-ranking | Week 9 | GPT-4o / Claude Sonnet | ___ |
| D16 | Does 3B (LLM rerank) outperform 3A? | After Phase 3 | Yes / No | ___ |
| D17 | Decay factor for no-doc folders in 3A | Week 9 | 0.80 / 0.90 / 0.95 | ___ |

---

## LLM Cost and Caching Notes

### Caching Strategy
- All LLM calls cached on disk keyed by `SHA256(prompt_text)`
- Folders without docs → augmented once, reused across all ECFs (saves ~50% of 2A/2C calls)
- Folders with docs → cached by `SHA256(folder_id + sorted(doc_ids))` — same doc set = same cache hit
- Phase 1 query generation: 45 topics × 4 types = **180 calls** (very cheap; run once)

### Estimated Call Volumes
| Component | Estimated LLM Calls | Notes |
|-----------|-------------------|-------|
| Phase 1 query gen | 180 | Fixed, cheap |
| Phase 2A augmentation | ~1,336 × 2 (doc + nodoc) × ECF variation (with caching ~40% reduction) | Largest cost |
| Phase 2B augmentation | ~668 empty folders × Homo prompt (with caching) | Less than 2A |
| Phase 2C augmentation | ~1,336 × 2 passes × ECF variation (with caching) | 2× Phase 2A |
| Phase 3B re-ranking | 45 topics × 30 ECFs × 1 call per k folders | Depends on k |

**Recommendation:** Run Phase 2A pilot on 5 ECFs first to estimate actual call count with caching, then decide if you need to reduce prompt complexity or use a cheaper model (e.g., GPT-4o-mini) for Phase 2C.

### Ablation Studies to Plan (within Phase 2A)
- DESCRIPTION only vs KEYWORDS only vs both in augmented label
- Original folder label preserved in index vs replaced entirely
- Top-3 docs vs top-5 docs vs all available docs per folder
- These can be run as sub-experiments within Phase 2A before scaling to all 30 ECFs