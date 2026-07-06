# Experiment Workbook: Folder Ranking in Sparsely Digitized Archives (Revised)

**SUSHI Collection — Victor Hugo Oliveira Leão, 2026**

> This document is a revised, living reference incorporating data-driven insights from QRELs analysis. Fill in decisions and results as experiments progress. Every `☐` is a task; every `___` is a value to fill after running experiments.
>
> **Revision Note (July 2026):** This revision incorporates three core insights from studying the relevant documents and folders for each topic:
>
> 1. Queries are generic terms/concepts, not specific events → expansion must focus on concept-bridging
> 2. SNC codes have bimodal specificity (specific like AGR vs generic like POL) → prompts must adapt
> 3. Document title + summary is sufficient context; documents are *critical* for generic SNC folders

---

## Data Schema Reference

This section documents the exact field names in each JSON data file and how they map to prompt variables and index configurations throughout this workbook.

### Folder Metadata (`data/folders_metadata/FoldersV1.3.json`)

The JSON file uses **folder IDs** as keys (e.g., `"A99990001"`). Each folder object has:

| JSON Field                | Type   | Always Filled?     | Description                                                                                                                                                                | Example                                                         |
| ------------------------- | ------ | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `box`                   | string | ✅ Yes             | Physical box identifier                                                                                                                                                    | `"A0001"`                                                     |
| `snc`                   | string | ✅ Yes             | Subject-Numeric Code (or`"Unknown"` for 68 folders)                                                                                                                      | `"POL 18"`, `"Unknown"`                                     |
| `label`                 | string | ✅**Always** | Original folder label — short descriptor with SNC + location + date.**This is always populated** and serves as the primary fallback when other fields are empty     | `"POL 15-1 BRAZ 01/01/1964"`, `"Economic Affairs(General)"` |
| `date`                  | string | ✅ Yes             | Start date (MM/DD/YYYY)                                                                                                                                                    | `"01/01/1964"`                                                |
| `endDate`               | string | ✅ Yes             | End date or`"Unknown"` (75.4% are Unknown)                                                                                                                               | `"01/01/1969"`, `"Unknown"`                                 |
| `rg`                    | string | ✅ Yes             | Record Group number                                                                                                                                                        | `"84"`                                                        |
| `label1965`             | string | Mostly             | SNC label from 1965 classification                                                                                                                                         | `"ORGANIZATIONS & CONFERENCES"`                               |
| `label1963`             | string | Fallback           | SNC label from 1963 classification (used when 1965 is missing)                                                                                                             | `"nan"` or actual label                                       |
| `raw_scope`             | string | 37.7%              | Full scope note text (empty string for 62.3% of folders)                                                                                                                   | Long text or`""`                                              |
| `scope_truncated`       | string | 37.7%              | Scope note truncated at stopper keywords (SEE, Exclude, etc.)                                                                                                              | Cleaned text or`""`                                           |
| `main_title`            | string | ✅**Always** | Resolved primary label (prefers label1965, falls back to label1963).**Always populated** — use as fallback when `label_parent_expanded` is empty or uninformative | `"ORGANIZATIONS & CONFERENCES"`                               |
| `parent`                | string | Partial            | Parent SNC code.`"None"` for top-level SNCs, empty for Unknown-SNC folders                                                                                               | `"POL"`, `"POL 13"`, `"None"`                             |
| `label_parent_expanded` | string | 85.8%              | Full hierarchical semantic path built by concatenating parent labels.**Empty for 190 folders** (14.2%) — all Unknown-SNC + some top-level SNCs                      | `"LABOR & MANPOWER: ORGANIZATIONS & CONFERENCES"`             |

> **Critical: `label` vs `label_parent_expanded` vs `main_title`**
>
> - `label` = the raw folder label as written on the physical folder. Always populated. For Unknown-SNC folders, contains descriptors like `"Economic Affairs(General)"`, `"Intelligence"`, `"Defense"`. **For known-SNC folders, contains the SNC code + country + date** (e.g., `"POL 15-1 BRAZ 01/01/1964"`).
> - `label_parent_expanded` = the **semantic hierarchy** built from SNC translations. Multi-level SNCs get rich expansions (e.g., `"POLITICAL AFFAIRS & RELATIONS: PROVINCIAL, MUNICIPAL & STATE GOVERNMENT"`). Top-level SNCs just repeat the name (e.g., `"AID"` → `"AID"`). **Empty for 190 folders.**
> - `main_title` = the SNC code's own human-readable meaning (e.g., `"POL 18"` → `"PROVINCIAL, MUNICIPAL & STATE GOVERNMENT"`). Always populated. **Use as the primary fallback** when `label_parent_expanded` is empty or equals the SNC code.

### Document Metadata (`data/items_metadata/itemsV1.2.json`)

The JSON file uses **document IDs** as keys (e.g., `"S01501"`). Each document object has:

| JSON Field            | Type                   | Description                                                    |
| --------------------- | ---------------------- | -------------------------------------------------------------- |
| `Sushi Box`         | string                 | Box identifier (links to folder's`box`)                      |
| `Sushi Folder`      | string                 | Folder identifier (links to folder key in FoldersV1.3.json)    |
| `Sushi File`        | string                 | Document filename (e.g.,`"S01501.pdf"`)                      |
| `title`             | string                 | Resolved document title                                        |
| `date`              | string                 | Document date (YYYY-MM-DD)                                     |
| `ocr`               | **list[string]** | OCR text**per page** — access `ocr[0]` for first page |
| `summary`           | string                 | GPT-4o generated summary of the document                       |
| `Brown Folder Name` | string                 | Brown University folder name (matches folder`label`)         |

### ECF File Format (`ecf/random_generated/`)

ECFs define which documents are "digitized" per experiment:

```json
{
  "ExperimentName": "ECF uniform w/ Random Seed 42",
  "ExperimentSets": [{
    "TrainingDocuments": ["BoxID/FolderID/SFile.pdf", ...],
    "Topics": {"T18Eval-00001": {"TITLE": "...", "DESCRIPTION": "...", ...}, ...}
  }]
}
```

- 5 Uniform ECFs: `ECF_RANDOM_{seed}.json` (~628 docs each, 5 docs/box)
- 1 All-docs ECF: `ECF_ALL_TRAINING_SET.json` (31,681 docs)
- To extract document ID from path: `path[-10:-4]` → e.g., `"S01501"`

### Topics Format (`src/data_creation/topics_output.txt`)

45 topics in JSON. Each topic has: `TITLE`, `DESCRIPTION`, `NARRATIVE`, `ID` (e.g., `"T18Eval-00001"`).
The experiments use `TITLE` → `{title}` and `DESCRIPTION` → `{description}` in prompts.

### QRELs Format (`qrels/formal-folder-qrel.txt`)

Tab-separated: `TopicID \t 0 \t FolderID \t RelevanceGrade`

- Relevance grades: 3 = Highly Relevant, 1 = Somewhat Relevant, 0 = Not Relevant
- Topic IDs format: `T18Eval-00001` through `T18Eval-00045`

### Prompt Variable → JSON Field Mapping

This mapping is used by ALL augmentation prompts (P2A-DOC, P2A-NODOC, P2B-HOMO, P2C-*):

| Prompt Variable            | JSON Source Field                                                              | Fallback Logic                                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `{folder_label}`         | `folder['label']`                                                            | Always populated — no fallback needed                                                                                    |
| `{snc}`                  | `folder['snc']`                                                              | If`"Unknown"` → display as `"Unclassified"`                                                                          |
| `{expanded_snc}`         | `folder['label_parent_expanded']`                                            | If empty or equals`folder['snc']` → use `folder['main_title']`. If SNC is Unknown → derive from `folder['label']` |
| `{parent_expanded_snc}`  | `folder['label_parent_expanded']` split at first `:` (parent portion only) | If empty → use`folder['main_title']`                                                                                   |
| `{scope_note}`           | `folder['scope_truncated']` (preferred) or `folder['raw_scope']`           | If empty string →`"No scope note available for this SNC code."`                                                        |
| `{start_date}`           | `folder['date']`                                                             | Always populated                                                                                                          |
| `{end_date}`             | `folder['endDate']`                                                          | If`"Unknown"` → handled in date formatting                                                                             |
| `{date_range_formatted}` | Computed                                                                       | `f"{start_date} to {end_date}"` if end_date known, else `f"{start_date} (end date not recorded)"`                     |
| `{record_group}`         | `folder['rg']`                                                               | Always populated                                                                                                          |
| `{main_title}`           | `folder['main_title']`                                                       | Always populated                                                                                                          |
| `{match_level}`          | Computed at runtime                                                            | `"exact"`, `"2-level"`, or `"parent"` (Phase 2B only)                                                               |
| `{snc_2level}`           | Computed: first two parts of SNC                                               | e.g.,`"POL 18"` → `"POL 18"`, `"POL 18-2"` → `"POL 18"`                                                         |
| `{parent}`               | `folder['parent']`                                                           | `"None"` for top-level SNCs                                                                                             |

```python
# Reference implementation for field resolution:
def resolve_folder_fields(folder: dict) -> dict:
    """Resolves all prompt variables from a folder metadata dict."""
    snc = folder['snc']
    label = folder['label']  # ALWAYS populated
    main_title = folder['main_title']  # ALWAYS populated
    label_parent_expanded = folder.get('label_parent_expanded', '')
  
    # SNC display
    snc_display = 'Unclassified' if snc == 'Unknown' else snc
  
    # Expanded SNC: prefer label_parent_expanded, fall back to main_title, then label
    if label_parent_expanded and label_parent_expanded != snc:
        expanded_snc = label_parent_expanded
    elif main_title and main_title != 'nan':
        expanded_snc = main_title
    else:
        expanded_snc = label  # Ultimate fallback — always populated
  
    # Parent expanded: broader category
    if label_parent_expanded and ':' in label_parent_expanded:
        parent_expanded = label_parent_expanded.split(':')[0].strip()
    elif main_title and main_title != 'nan':
        parent_expanded = main_title
    else:
        parent_expanded = label
  
    # Scope note
    scope = folder.get('scope_truncated', '') or folder.get('raw_scope', '')
    scope_display = scope if scope else 'No scope note available for this SNC code.'
  
    # Date range
    start_date = folder['date']
    end_date = folder['endDate']
    if end_date == 'Unknown':
        date_range = f"{start_date} (end date not recorded)"
    else:
        date_range = f"{start_date} to {end_date}"
  
    # SNC hierarchy for cascading homophily
    parts = snc.split('-')[0].split()  # "POL 18-2" → ["POL", "18"]
    snc_2level = ' '.join(parts[:2]) if len(parts) >= 2 else snc
    parent = folder.get('parent', 'None')
  
    return {
        'folder_label': label,
        'snc': snc_display,
        'expanded_snc': expanded_snc,
        'parent_expanded_snc': parent_expanded,
        'scope_note': scope_display,
        'start_date': start_date,
        'end_date': end_date,
        'date_range_formatted': date_range,
        'record_group': folder['rg'],
        'main_title': main_title,
        'snc_2level': snc_2level,
        'parent': parent,
    }
```

---

## Reference Guide

### Retrieval Models

| Code | Model                                   | Notes                                                                               |
| ---- | --------------------------------------- | ----------------------------------------------------------------------------------- |
| B    | BM25F                                   | Tuned. For AllF phase, retune for folder fields (not TOFS). Default: k1=1.5, b=0.75 |
| E    | Sentence Embeddings (all-mpnet-base-v2) | Dense bi-encoder                                                                    |
| C    | ColBERTv2 (PLAID)                       | Late interaction                                                                    |
| BCE  | RRF(B + C + E, k=60)                    | Primary ensemble                                                                    |
| BC   | RRF(B + C)                              | —                                                                                  |
| BE   | RRF(B + E)                              | —                                                                                  |
| CE   | RRF(C + E)                              | —                                                                                  |

> **Existing Implementation Reference:**
> The retrieval models are already implemented in [`src/models.py`](file:///home/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/models.py):
>
> - `BM25Model` — PyTerrier-based, auto-switches BM25↔BM25F based on field count. Field weights: title(w=3.0,c=0.5), ocr(w=0.5,c=0.4), folderlabel(w=1.3,c=0.65), summary(w=1.0,c=1.5). **For Phase 1 AllF experiments, these weights need retuning for folder-level fields.**
> - `EmbeddingsModel` — SentenceTransformers (`all-mpnet-base-v2`), cosine similarity.
> - `ColBERTModel` — pylate/ColBERT (`lightonai/colbertv2.0`), PLAID indexing.
> - All implement `train(data: list[dict])` and `search(query: str) -> pd.DataFrame`.
> - RRF fusion is implemented in [`src/hybrid_models.py`](file:///home/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/hybrid_models.py) via `perform_hybrid_fusion()`.

### Masking Conditions

| Code | Description                             | ECFs | Notes                   |
| ---- | --------------------------------------- | ---- | ----------------------- |
| Uni  | Random Uniform (5 docs/box, ~630 total) | 5    | Mean ± 95% CI reported |
| NoM  | No Mask (all 31,681 docs)               | 1    | Upper bound only        |

### Evaluation

- **Metric:** Folder nDCG@5, averaged over 45 topics
- **Reporting:** Mean ± 95% CI
- **Significance:** Wilcoxon signed-rank test, paired by topic, p < 0.05. Mark significant results with *
- **Baseline to beat:** TOFS + BCE + AllF + SimSNC n=2 → **0.2141 ± 0.049 (Uni)**

> **Existing Implementation Reference:**
> Evaluation is implemented in [`src/evaluator.py`](file:///home/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/evaluator.py):
>
> - `Evaluator.evaluate(run_file_path, output_json_path)` — uses `pytrec_eval` for nDCG@5, MAP, MRR, plus custom `count_relevant_top5`.
> - `Evaluator.generate_aggregated_metrics(folder_path, run_type)` — aggregates over 5 seeds: mean + 95% CI (t-distribution).
> - `Evaluator.save_run_file(results, output_path, run_name)` — TREC format: `query_id \t doc_id \t rank \t score \t run_name`.
> - QRELs path: `qrels/formal-folder-qrel.txt` (tab-separated, relevance grades 0/1/3).

### SNC Specificity Classification (NEW — Data-Driven)

> **Key Insight:** SNC codes have a bimodal specificity distribution. Some directly name their thematic content (AGR = Agriculture) while others are too broad to predict specific content (POL = 561 folders covering dozens of sub-topics). The augmentation strategy MUST differ based on this classification.

| Category           | Definition                                                | SNC Parents                                                                                                                                          | Folder Count | Expansion Strategy                                                         |
| ------------------ | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | -------------------------------------------------------------------------- |
| **Specific** | The SNC label directly tells you what the folder contains | AGR, LAB, HLTH, DEF, SCI, EDX, EDU, AID, PET, TEL, SP, AV, CSM, REF, INCO, FN, TP, PRM, INT, CUL, FT, LEG, INF, CR, BUD, and many single-folder SNCs | ~600 folders | Expand with**related concepts and terms** — the label is the anchor |
| **Generic**  | The SNC label is too broad to predict specific content    | POL (561), E (29), SOC (38), ORG (34), PER (31), Unknown (68)                                                                                        | ~761 folders | Expand with**document-derived context** — documents are essential   |

```python
# Implementation: SNC specificity classifier
GENERIC_SNC_PARENTS = {'POL', 'E', 'SOC', 'ORG', 'PER', 'Unknown'}

def classify_snc_specificity(snc: str) -> str:
    """Classify SNC as 'specific' or 'generic'."""
    if snc == 'Unknown':
        return 'generic'
    parent = snc.split()[0]
    if parent in GENERIC_SNC_PARENTS:
        return 'generic'
    return 'specific'
```

### Folder Index Configurations (Phase 1)

| Code | Content indexed per folder                                                                                               |
| ---- | ------------------------------------------------------------------------------------------------------------------------ |
| F1   | Folder Label only                                                                                                        |
| F2   | Folder Label + Parent Expanded SNC (uses`main_title` as fallback when `label_parent_expanded` is empty)              |
| F3   | Folder Label + Scope Note (blank for folders without scope note)                                                         |
| F4   | Folder Label + Parent Expanded SNC + Scope Note (uses`main_title` as fallback when `label_parent_expanded` is empty) |
| F5   | Folder Label + Parent Expanded SNC + Scope Note +`main_title` (always populated)                                       |

> **Data-Driven Note — Unknown SNC Handling (Phase 0 Finding):**
> 68 folders (5.1%) have `SNC="Unknown"` with empty `label_parent_expanded` and empty `parent`.
> These contain **1,328 documents** (4.2% of collection) with meaningful labels like
> *"Economic Affairs(General)"*, *"Intelligence"*, *"Education 1966"*, *"Defense"*.
> For these folders:
>
> 1. F2/F4/F5 configs: use `label` + `main_title` as the expanded SNC substitute
> 2. Augmentation prompts: derive SNC meaning from the folder label text
> 3. Homophily: match by `main_title` similarity or `box` proximity instead of SNC
>    This ensures these 1,328 documents remain findable.

> **Data-Driven Note — `label_parent_expanded` Quality (Phase 0 Finding):**
>
> - 190 folders (14.2%) have empty `label_parent_expanded` (all Unknown-SNC + some top-level)
> - Top-level SNCs (AID, DEF, FN, etc.) have minimal expansion — e.g., `"AID"` → `"AID"`
> - Multi-level SNCs have rich expansions — e.g., `"POL 18"` → `"POLITICAL AFFAIRS & RELATIONS: PROVINCIAL, MUNICIPAL & STATE GOVERNMENT"`
> - **Implication for F2/F4:** Most benefit comes from multi-level SNCs. For top-level and Unknown SNCs, F2 ≈ F1.
> - **Implementation:** When `label_parent_expanded` is empty or equals the SNC code, fall back to `main_title`.

> **Implementation Guide — Building Folder Index Configs:**
>
> The existing codebase uses `label_parent_expanded` as `folderlabel` in [`src/run_generator.py`](file:///home/victorleao/mestrado/SUSHI_Information_Retrieval_Archives/src/run_generator.py) (lines 206-219). For the new experiments, each F-config must be built differently:
>
> ```python
> def build_folder_index_text(folder: dict, config: str) -> str:
>     """Builds the indexable text for a folder based on the F-config."""
>     label = folder['label']  # ALWAYS populated
>     main_title = folder['main_title']  # ALWAYS populated
>     lpe = folder.get('label_parent_expanded', '')
>     scope = folder.get('scope_truncated', '') or folder.get('raw_scope', '')
>   
>     # Resolve parent_expanded: use main_title as fallback
>     parent_expanded = lpe if (lpe and lpe != folder['snc']) else main_title
>   
>     if config == 'F1':
>         return label
>     elif config == 'F2':
>         return f"{label} | {parent_expanded}"
>     elif config == 'F3':
>         if scope:
>             return f"{label} | {scope}"
>         return label  # F3 degrades to F1 when no scope note
>     elif config == 'F4':
>         parts = [label, parent_expanded]
>         if scope:
>             parts.append(scope)
>         return ' | '.join(parts)
>     elif config == 'F5':
>         parts = [label, parent_expanded]
>         if scope:
>             parts.append(scope)
>         parts.append(main_title)
>         return ' | '.join(parts)
> ```
>
> **Training data format** (adapting from existing `prepare_training_data()`):
>
> ```python
> # For each folder in FoldersV1.3.json:
> training_entry = {
>     'docno': folder_id,       # Folder ID (e.g., 'A99990001')
>     'folder': folder_id,      # Same as docno for folder-level indexing
>     'box': folder['box'],
>     'date': folder['date'],
>     'folderlabel': build_folder_index_text(folder, config),  # F1-F5
>     'text_blob': build_folder_index_text(folder, config),    # For dense models
> }
> ```

### Query Types (Original + Revised)

#### Original Query Types (Phase 1)

| Code    | Description                                            | Use with                                  |
| ------- | ------------------------------------------------------ | ----------------------------------------- |
| Q0      | Normal TD (Title + Description)                        | All models                                |
| Q1      | HyDE — LLM generates SNC-style folder label           | All models                                |
| Q2      | Keywords — entities, events, institutions list        | **B, BC, BCE only** (never E alone) |
| Q3      | Embedding Text — descriptive historical paragraph     | E, CE, BE, BCE                            |
| Q4      | Reformulations — 4 novel queries (no Q0 inside)       | All models                                |
| Q0+Q1   | Real query augmented with HyDE labels                  | B, E, BCE                                 |
| Q0+Q2   | Real query augmented with Keywords                     | B, BCE                                    |
| Q0+Q3   | Real query augmented with Embedding Text               | E, BCE                                    |
| Q0+Q4   | Real query augmented with Reformulations               | B, E, BCE                                 |
| Q0+QALL | Real query augmented with all 4 LLM expansions         | B, E, BCE                                 |
| QALL    | All 4 LLM expansions concatenated (Q1+Q2+Q3+Q4, no Q0) | BCE                                       |
| Q-TCDE  | Phase 1 query × 3 + 3 LLM topic passages              | BCE (Phase 2C only)                       |

#### Revised Query Types (Phase 1B-R — NEW)

> **Design Rationale for Revised Queries:**
> The original Q1-Q3 expansions were designed around *events, entities, and key figures*. Data analysis reveals that queries are mostly generic topical terms (e.g., "coffee", "submarines", "floods") and folder labels are thematic codes (e.g., "AGRICULTURE", "DEFENSE AFFAIRS"). The revised queries target the **concept-to-category gap** rather than the **modern-to-historical vocabulary gap**.

| Code      | Description                                                                    | Use with       | Replaces |
| --------- | ------------------------------------------------------------------------------ | -------------- | -------- |
| Q5        | Concept Bridge — related concepts at 3 levels (categories, sub-topics, terms) | All models     | Q2 role  |
| Q6        | SNC-Aware HyDE — folder descriptions in real SNC label style                  | All models     | Q1       |
| Q7        | Thematic Paragraph — concept/term-focused (not event-focused)                 | E, CE, BE, BCE | Q3       |
| Q2R       | Term-Focused Keywords — domain concepts, not people names                     | B, BC, BCE     | Q2       |
| Q0+Q5     | Original + Concept Bridge                                                      | B, E, BCE      | Q0+Q2    |
| Q0+Q6     | Original + SNC-Aware HyDE                                                      | B, E, BCE      | Q0+Q1    |
| Q0+Q5+Q6  | Original + Concept + HyDE                                                      | BCE            | —       |
| Q0+QALL-R | Original + Q5 + Q6 + Q7 + Q2R                                                  | B, E, BCE      | Q0+QALL  |

---

## Phase 0: Data Analysis

### Why

The Homophily augmentation strategy (Phase 2B) requires knowing which SNC codes have usable neighbor documents. Some SNCs may be "digitization deserts" where no training docs are available, making homophily infeasible. Phase 0 prevents wasted effort in Phase 2B and sets realistic expectations.

### ☐ Checklist

**0.1 SNC Distribution**

- [X] Load `folder_metadata.json` for all 1,336 folders
- [X] Count folders per 3-level SNC code (e.g., "POL 18"), per 2-level (e.g., "POL"), per 1-level primary code
- [X] Plot histogram of folders per SNC
- [X] List top-10 and bottom-10 SNCs by folder count
- [X] Note: 379 distinct SNCs, **57** have scope notes — flag which SNCs benefit from F3/F4
- [X] Distribution of documents to SNC (see if there are more favored SNC)
- [X] **Finding:** POL parent dominates with 353 folders (26.4%), 11,432 docs. 68 folders have SNC="Unknown".

**0.2 Digitization Coverage per SNC**

- [X] Load one representative Uniform ECF (ECF_RANDOM_42: 628 training docs, 562 unique folders)
- [X] For each SNC, compute: (folders with ≥1 training doc) / (total folders with that SNC)
- [X] Classify each SNC:
  - **Rich:** ≥50% of folders have training docs
  - **Moderate:** 20–49% have training docs
  - **Poor:** <20% have training docs
- [X] Identify SNCs where >80% of folders are empty in Uniform ECFs (homophily won't help)

**0.3 Homophily Feasibility Study**

- [X] For each folder without training docs (representative Uni ECF, 774 empty folders), count:
  - Same-SNC (exact) folders that DO have training docs → **69.9% ≥1 doc, 43.0% ≥3 docs**
  - Same-SNC (2-level prefix, e.g., "POL 23") folders with training docs → **76.9% ≥1 doc, 55.0% ≥3 docs**
  - Parent-level SNC folders with training docs → **89.9% ≥1 doc, 73.8% ≥3 docs**
- [X] Compute distribution: "# of same-SNC docs available for the average empty folder"
- [X] Answer: 73.8% of empty folders have ≥3 parent-level neighbor docs (viability threshold: ≥20%)
- [X] **Decision D6 = Cascading hierarchy** (exact → 2-level → parent). **Decision D9 = Yes** (73.8% viability).

**0.4 Scope Note Coverage**

- [X] Count: **504 of 1,336 folders** (37.7%) have non-empty scope notes, from **57 of 379 distinct SNCs** (15%)
- [X] List which SNC primary codes have scope notes (to predict F3/F4 benefit)
- [X] Note this affects Phase 1 Set 1A: F3 and F4 are sparse — **62.3% of folders have no scope note**

**0.5 Date Range Summary**

- [X] Compute distribution of folder date ranges (year of start date) — all 1,336 have valid start dates
- [X] Flag folders with missing or anomalous dates: **1,008 folders (75.4%) have endDate="Unknown"**
- [X] **All prompts must handle Unknown end dates gracefully** — this is the majority of folders

**0.6 Textual SNC**

- [X] Check the different SNC values in order to see which have null values.
- [X] **Finding:** 68 folders have SNC="Unknown", 190 folders have empty `label_parent_expanded`
- [X] **Finding:** Top-level SNCs have minimal expansion (e.g., "AID" → "AID"), while sub-level SNCs have rich expansions

**0.7 Documents**

- [X] Analyze some documents Title, Summary and OCR inside the SNC folders to see if they make sense.
- [X] **Finding:** Document titles and summaries are meaningful and align well with SNC classifications

**0.8 SNC Specificity Analysis (NEW)**

- [X] Classify all SNCs as specific or generic based on parent code
- [X] **Finding:** POL dominates generic category (561/761 generic folders). Specific SNCs like AGR, DEF, LAB have clear thematic identity that directly matches query terms.
- [X] **Finding:** For specific SNCs, the expansion must bridge concepts (e.g., AGR → coffee, cocoa, crop production). For generic SNCs, documents are the primary signal for folder content.

### Required Outputs from Phase 0

1. SNC coverage table: `SNC → folder_count → avg_training_docs_per_ECF → digitization_category`
2. Homophily viability number: `% of empty folders with ≥{D7} same-SNC docs` → **73.8% at parent level (threshold 3)**
3. Filled decisions D6, D7, D8, D9 (see Future Decisions Tracker)
4. Insights about the SNCs and how they work.
5. Insights about the documents and their relation to the SNC.
6. **Date completeness report:** 75.4% of folders lack end dates — prompts must handle gracefully.
7. **Unknown SNC report:** 68 folders (5.1%) with 1,328 docs have no SNC classification — use `label` fallback.
8. **SNC specificity classification:** specific vs generic for all 1,336 folders.

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

### Why Each Revised Query Type Works (NEW)

**Q5 — Concept Bridge (NEW)**
The core problem: queries use one vocabulary ("coffee"), folder labels use another ("AGRICULTURE"). Q5 explicitly generates multi-level concept bridges: broad categories → sub-topics → concrete terms. This creates vocabulary overlap at every level of the SNC hierarchy, not just at the event/entity level. Effective for both BM25 (more matching terms) and dense models (richer semantic signal).

**Q6 — SNC-Aware HyDE (replaces Q1)**
Original Q1 generates fake SNC codes ("SCI 12-4 BRAZ") that don't exist in the index. Q6 generates *thematic descriptions in the style of real SNC labels* (e.g., "AGRICULTURE: COMMODITY EXPORTS AND TRADE POLICY"). This produces vocabulary that actually overlaps with the indexed folder label text. Uses real SNC label examples to calibrate output style.

**Q7 — Thematic Paragraph (replaces Q3)**
Original Q3 over-focuses on specific historical events and diplomatic actors. Q7 is concept-focused: it describes the thematic scope, related sub-topics, and domain terms rather than narrating events. Better for semantic matching because it captures the *concept space* of the query rather than a specific historical narrative.

**Q2R — Term-Focused Keywords (replaces Q2)**
Original Q2 generates proper nouns ("Ernesto Geisel, ARENA party") — terms that rarely appear in folder labels. Q2R generates *thematic terms* ("land reform, military governance, agricultural policy") that match the vocabulary used in SNC descriptions and folder labels. Includes SNC parent code abbreviations (POL, AGR, DEF) as explicit bridge terms.

---

### Code to Implement

```
src/query_expansion/
├── generate_hyde.py           # LLM → Q1 for all 45 topics → save queries_hyde.json
├── generate_keywords.py       # LLM → Q2 for all 45 topics → save queries_keywords.json
├── generate_emb_text.py       # LLM → Q3 for all 45 topics → save queries_embtext.json
├── generate_reformulations.py # LLM → Q4 for all 45 topics → save queries_reformulations.json
├── generate_concept.py        # LLM → Q5 for all 45 topics → save queries_concept.json (NEW)
├── generate_hyde_r.py         # LLM → Q6 for all 45 topics → save queries_hyde_r.json (NEW)
├── generate_thematic.py       # LLM → Q7 for all 45 topics → save queries_thematic.json (NEW)
├── generate_keywords_r.py     # LLM → Q2R for all 45 topics → save queries_keywords_r.json (NEW)
├── combine_queries.py         # Combines saved outputs → Q12, Q13, Q14, Q-TCDE
└── combine_queries_r.py       # Combines revised outputs → Q0+Q5, Q0+Q6, etc. (NEW)

src/retrieval/
├── allf_index.py              # Builds index from folder metadata (F1–F5 configs)
├── allf_retrieval.py          # Runs retrieval: given query + index + model → ranked list
└── results_manager.py         # Saves ranked lists with metadata for later scoring

src/utils/
├── llm_client.py              # LLM API wrapper with SHA256-based disk caching
├── field_resolver.py          # resolve_folder_fields() — maps JSON fields to prompt variables
├── snc_classifier.py          # classify_snc_specificity() — specific vs generic (NEW)
└── ecf_utils.py               # ECF loading + training doc extraction per folder
```

**LLM Caching:** Cache by `(topic_id, query_type)`. All 45 topics × 8 types (4 original + 4 revised) = **360 LLM calls** total for Phase 1 query generation. Run once, store JSON, reuse forever.

> **Implementation Guide — LLM Client (`src/utils/llm_client.py`):**
>
> This utility wraps OpenAI/Groq API calls with deterministic disk caching:
>
> ```python
> import hashlib, json, os
> from openai import OpenAI  # or from groq import Groq
>
> class LLMClient:
>     def __init__(self, cache_dir: str, model: str = 'gpt-4o', provider: str = 'openai'):
>         self.cache_dir = cache_dir
>         os.makedirs(cache_dir, exist_ok=True)
>         self.model = model
>         if provider == 'openai':
>             self.client = OpenAI()
>         elif provider == 'groq':
>             from groq import Groq
>             self.client = Groq()
>
>     def _cache_key(self, prompt: str) -> str:
>         return hashlib.sha256(prompt.encode()).hexdigest()
>
>     def generate(self, prompt: str, system_prompt: str = '', temperature: float = 0.3) -> str:
>         key = self._cache_key(system_prompt + prompt)
>         cache_path = os.path.join(self.cache_dir, f"{key}.json")
>         if os.path.exists(cache_path):
>             with open(cache_path, 'r') as f:
>                 return json.load(f)['response']
>   
>         messages = []
>         if system_prompt:
>             messages.append({'role': 'system', 'content': system_prompt})
>         messages.append({'role': 'user', 'content': prompt})
>   
>         response = self.client.chat.completions.create(
>             model=self.model, messages=messages, temperature=temperature
>         )
>         text = response.choices[0].message.content
>   
>         with open(cache_path, 'w') as f:
>             json.dump({'prompt_hash': key, 'response': text, 'model': self.model}, f)
>         return text
> ```

---

### Prompts — Original (Phase 1)

---

#### PROMPT P1-HYDE (Original)

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

#### PROMPT P1-KW (Original)

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

#### PROMPT P1-EMB (Original)

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

#### PROMPT P1-REF (Original)

*Goal:* Generate 4 complementary reformulations to capture different facets of the query.

```
You are an expert in archival research on U.S. State Department records on Brazil (1960s-1970s). Your task is to expand the following research topic by generating 4 NOVEL and DISTINCT search queries. DO NOT just copy or slightly reword the provided title and description. You must use your historical knowledge to infer related concepts, specific names, and events that are not explicitly mentioned but are highly relevant to the topic.

Research Topic Title: {title}
Research Topic Description: {description}

Generate 4 queries, each approaching the topic from a different angle:
Query 1 (U.S. Perspective): Focus on the official U.S. government perspective, actions, policies, or specific agencies involved.
Query 2 (Brazilian Context): Focus on Brazilian actors, institutions, domestic politics, or local context.
Query 3 (Specific Events/Policies): Focus on specific historical events, operations, incidents, or policies related to the topic.
Query 4 (Key Entities): Focus on key named individuals, groups, or organizations.

Rules:
- CRITICAL: Do NOT simply repeat phrases from the title or description. Generate NEW vocabulary and terms related to the topic.
- Each query must be 5-15 words long.
- Maximize the diversity of vocabulary across the 4 queries.
- Return ONLY the 4 queries, one per line, without numbering, labels, or explanations.
```

*Post-processing:* Final Q4 string = `{query1} {query2} {query3} {query4}` (only the LLM-generated reformulations; Q0 augmentation is handled separately via Q0+Q4).

---

### Prompts — Revised (Phase 1B-R — NEW)

---

#### PROMPT P1R-CONCEPT (Q5 — Concept Bridge)

*Goal:* Generate a multi-level list of related concepts and terms that bridge the gap between the user's query vocabulary and the SNC/folder label vocabulary.

*Why this works:* The user searches for "coffee" but the folder is labeled "AGR Agriculture 1964". The concept bridge adds "agriculture, agricultural commodities, coffee exports, crop production, soil conservation" — terms that overlap with both the query intent and the folder label vocabulary.

```
You are an expert on U.S. State Department archival classification. The archive uses Subject-Numeric Codes (SNC) to organize folders on Brazil (1960s-1970s). 

Here are examples of REAL SNC category labels from the collection:
- AGRICULTURE
- POLITICAL AFFAIRS & RELATIONS
- LABOR & MANPOWER
- SCIENCE & TECHNOLOGY
- DEFENSE AFFAIRS
- ECONOMIC AFFAIRS
- HEALTH & SANITATION
- EDUCATION & CULTURE
- SOCIAL CONDITIONS
- HEAD OF STATE. EXECUTIVE BRANCH.
- GOVERNMENT: PROVINCIAL, MUNICIPAL & STATE GOVERNMENT
- INTERNAL SECURITY & INTELLIGENCE

Given the following research topic, generate a list of RELATED CONCEPTS AND TERMS that would help match this topic to the correct archival folders. 

Research Topic Title: {title}
Research Topic Description: {description}

Generate terms at THREE levels:
1. BROAD CATEGORIES: Which general archival categories (like the examples above) would contain relevant folders? List 2-4.
2. SUB-TOPICS: What specific sub-topics within those categories relate to this query? List 5-8 terms.
3. CONCRETE TERMS: What specific terms, commodities, programs, or concepts would appear in documents filed under these folders? List 8-12 terms. Focus on terms relevant to Brazil in the 1960s-1970s.

Return three lines:
CATEGORIES: [comma-separated list]
SUB-TOPICS: [comma-separated list]  
TERMS: [comma-separated list]

No explanations, no numbering beyond the three line labels.
```

*Post-processing:* Remove "CATEGORIES:", "SUB-TOPICS:", "TERMS:" labels, concatenate all terms as Q5 string.

---

#### PROMPT P1R-HYDE (Q6 — SNC-Vocabulary-Aware HyDE)

*Goal:* Generate a hypothetical folder description using vocabulary drawn from real SNC labels, not invented SNC codes.

*Why this replaces Q1:* The original HyDE generates fake SNC codes (e.g., "SCI 12-4 BRAZ") that don't exist in the index. The revised version generates *descriptions in the style of real SNC labels*, which produces vocabulary that actually overlaps with the indexed folder text.

```
You are an expert on U.S. State Department archival classification for Brazil (1960s-1970s).
Folders in this archive are labeled with Subject-Numeric Codes (SNC) that describe their thematic content. Here are examples of REAL folder label descriptions:

- "AGRICULTURE" — covers crop production, agricultural policy, commodities
- "POLITICAL AFFAIRS & RELATIONS: HEAD OF STATE. EXECUTIVE BRANCH." — covers presidential actions, executive decisions
- "POLITICAL AFFAIRS & RELATIONS: GOVERNMENT: PROVINCIAL, MUNICIPAL & STATE GOVERNMENT" — covers regional politics
- "LABOR & MANPOWER: LABOR CONDITIONS & LABOR RELATIONS" — covers unions, strikes, working conditions
- "SCIENCE & TECHNOLOGY: TECHNOLOGICAL RESEARCH" — covers scientific programs, technology transfer
- "DEFENSE AFFAIRS: ARMED FORCES" — covers military operations, defense policy

Given the following research topic, write 2-3 SHORT folder descriptions (10-20 words each) in the SAME STYLE as the examples above. Describe what a matching folder would be about thematically — do NOT invent SNC codes or dates.

Research Topic Title: {title}
Research Topic Description: {description}

Return ONLY the folder descriptions, one per line. No codes, no dates, no explanations.
```

*Post-processing:* Concatenate all generated descriptions (space-separated) as the Q6 string.

---

#### PROMPT P1R-THEMATIC (Q7 — Thematic Paragraph)

*Goal:* Generate a rich thematic paragraph focused on concepts and terms rather than events and figures.

*Why this revises Q3:* The original Q3 generates good descriptive text but over-focuses on specific historical events and diplomatic actors. The revised version emphasizes *thematic concepts, related terminology, and the types of content* that would be in matching folders.

```
You are an expert on U.S.-Brazil diplomatic archives (1960s-1970s). Write a thematic description for the research topic below, optimized for matching against archival folder labels and descriptions.

Research Topic Title: {title}
Research Topic Description: {description}

Write a single paragraph of 100-150 words that:
- Describes the THEMATIC SCOPE of what this topic covers (not specific events)
- Lists related concepts, sub-topics, and domain terms that a researcher would associate with this topic
- Uses vocabulary that would appear in archival folder labels and classification systems
- Mentions the types of content (policy documents, reports, cables, assessments) that would be filed under this topic
- Includes related concepts that broaden the match (e.g., for "coffee" also mention "agricultural policy, commodity exports, trade agreements")

Focus on TERMS AND CONCEPTS, not on narrating specific events or naming specific people.

Write only the paragraph. No title, no header, no preamble.
```

*Post-processing:* Use directly as Q7 string.

---

#### PROMPT P1R-KW (Q2R — Term-Focused Keywords)

*Goal:* Same purpose as original Q2 but refocused from entities/figures to concepts/terms.

```
You are a research assistant specializing in archival classification systems. Generate a list of search terms optimized for matching against FOLDER LABELS (not document content) in a U.S. State Department archive on Brazil (1960s-1970s).

Research Topic Title: {title}
Research Topic Description: {description}

Focus your terms on:
- Thematic categories and sub-categories (e.g., "agriculture", "political affairs", "labor relations")
- Domain-specific concepts (e.g., "land reform", "military government", "economic development")  
- Related terms and synonyms that broaden the match (e.g., for "coffee" → "crops, commodities, exports, agricultural production")
- Brazilian-context terms relevant to the 1960s-1970s
- Abbreviations used in State Department classification (e.g., POL, AGR, DEF, ECON)

DO NOT focus on:
- Specific people's names (these rarely appear in folder labels)
- Specific event dates
- Geographic locations (unless the topic is specifically about a location)

Return ONLY a single comma-separated list. Maximum 25 terms.
```

*Post-processing:* Replace commas with spaces.

---

### Experiment Set 1A: Normal Queries × Folder Index Fields (UNCHANGED)

**Purpose:** Identify the best folder metadata configuration for AllF retrieval.
**Query:** Q0 (TD — Title + Description, same as baseline)
**Evaluation:** nDCG@5 over 45 topics, single run

| Run ID | Index Config                       | Model | Compare Against |
| ------ | ---------------------------------- | ----- | --------------- |
| 1A-01  | F1 (Label)                         | B     | —              |
| 1A-02  | F1                                 | E     | —              |
| 1A-03  | F1                                 | C     | —              |
| 1A-04  | F1                                 | BCE   | —              |
| 1A-05  | F1                                 | BC    | —              |
| 1A-06  | F1                                 | BE    | —              |
| 1A-07  | F1                                 | CE    | —              |
| 1A-08  | F2 (Label + ParentSNC)             | B     | 1A-01           |
| 1A-09  | F2                                 | E     | 1A-02           |
| 1A-10  | F2                                 | C     | 1A-03           |
| 1A-11  | F2                                 | BCE   | 1A-04           |
| 1A-12  | F2                                 | BC    | 1A-05           |
| 1A-13  | F2                                 | BE    | 1A-06           |
| 1A-14  | F2                                 | CE    | 1A-07           |
| 1A-15  | F3 (Label + ScopeNote)             | B     | 1A-01           |
| 1A-16  | F3                                 | E     | 1A-02           |
| 1A-17  | F3                                 | C     | 1A-03           |
| 1A-18  | F3                                 | BCE   | 1A-04           |
| 1A-19  | F3                                 | BC    | 1A-05           |
| 1A-20  | F3                                 | BE    | 1A-06           |
| 1A-21  | F3                                 | CE    | 1A-07           |
| 1A-22  | F4 (Label + ParentSNC + ScopeNote) | B     | 1A-01           |
| 1A-23  | F4                                 | E     | 1A-02           |
| 1A-24  | F4                                 | C     | 1A-03           |
| 1A-25  | F4                                 | BCE   | 1A-04           |
| 1A-26  | F4                                 | BC    | 1A-05           |
| 1A-27  | F4                                 | BE    | 1A-06           |
| 1A-28  | F4                                 | CE    | 1A-07           |

**Total 1A: 28 runs**

> **Note on F5:** F5 is defined as a configuration but is not included in Set 1A to keep the initial experiment manageable (28 runs). If F4 shows meaningful improvement over F1, test F5 as a follow-up with the best-performing model only (7 additional runs). If F2 ≈ F1 for all models, F5 will not add value either. Record this decision after reviewing 1A results.

☐ **After 1A:** Record F_best for B, E, C, BCE individually.

> **Data-Driven Note — Scope Notes (Phase 0 Finding):** Only 57/379 SNCs (15%) have scope notes,
> covering 504/1,336 folders (37.7%). F3/F4 may show modest gains overall but could significantly
> help the 37.7% of folders with scope notes. Consider computing nDCG@5 **separately** for
> folders with vs. without scope notes to measure targeted impact of F3/F4.

---

### Experiment Set 1B: Original Expanded Queries (UNCHANGED)

**Purpose:** Identify the best query strategy for AllF retrieval using original (entity/event-focused) expansions.
**Index:** F_best (best configuration per model from Set 1A)
**Evaluation:** nDCG@5 over 45 topics, single run

> **Experiment Design:**
> Set 1B is organized in layers of increasing complexity:
>
> 1. **1B-1 to 1B-4** — Individual LLM expansions alone (replaces Q0 entirely)
> 2. **1B-5** — Q0 augmented with individual LLM expansions (preserves original intent)
> 3. **1B-6** — Multi-expansion combinations (LLM-only vs Q0-anchored)
> 4. **1B-7** — Best query × all index configs (query-index interaction)

#### 1B-1: HyDE (Q1) — all models

| Run ID | Query | Model | Compare Against    |
| ------ | ----- | ----- | ------------------ |
| 1B-01  | Q1    | B     | 1A best B result   |
| 1B-02  | Q1    | E     | 1A best E result   |
| 1B-03  | Q1    | C     | 1A best C result   |
| 1B-04  | Q1    | BCE   | 1A best BCE result |
| 1B-05  | Q1    | BC    | 1A best BC result  |
| 1B-06  | Q1    | BE    | 1A best BE result  |
| 1B-07  | Q1    | CE    | 1A best CE result  |

#### 1B-2: BM25 Keywords (Q2) — B, BC, BCE only

| Run ID | Query | Model | Compare Against     |
| ------ | ----- | ----- | ------------------- |
| 1B-08  | Q2    | B     | 1A best B / 1B-01   |
| 1B-09  | Q2    | BC    | 1A best BC / 1B-05  |
| 1B-10  | Q2    | BCE   | 1A best BCE / 1B-04 |

#### 1B-3: Embedding Text (Q3) — E, CE, BE, BCE

| Run ID | Query | Model | Compare Against     |
| ------ | ----- | ----- | ------------------- |
| 1B-11  | Q3    | E     | 1A best E / 1B-02   |
| 1B-12  | Q3    | CE    | 1A best CE / 1B-07  |
| 1B-13  | Q3    | BE    | 1A best BE / 1B-06  |
| 1B-14  | Q3    | BCE   | 1A best BCE / 1B-04 |

#### 1B-4: Reformulations (Q4) — all models

| Run ID | Query | Model | Compare Against     |
| ------ | ----- | ----- | ------------------- |
| 1B-15  | Q4    | B     | 1A best B / 1B-01   |
| 1B-16  | Q4    | E     | 1A best E / 1B-02   |
| 1B-17  | Q4    | C     | 1A best C / 1B-03   |
| 1B-18  | Q4    | BCE   | 1A best BCE / 1B-04 |
| 1B-19  | Q4    | BC    | 1A best BC / 1B-05  |
| 1B-20  | Q4    | BE    | 1A best BE / 1B-06  |
| 1B-21  | Q4    | CE    | 1A best CE / 1B-07  |

#### 1B-5: Real Query + Individual LLM Expansion (Q0+Qx)

> **Rationale:** Sections 1B-1 to 1B-4 test LLM expansions in isolation, replacing Q0 entirely. This section tests whether *augmenting* the real query with each expansion yields better results than either alone. The original query anchors the user's intent while the LLM text adds archival vocabulary. Models tested are those best suited for each expansion type.

| Run ID | Query | Model | Compare Against     |
| ------ | ----- | ----- | ------------------- |
| 1B-22  | Q0+Q1 | B     | 1A best B / 1B-01   |
| 1B-23  | Q0+Q1 | E     | 1A best E / 1B-02   |
| 1B-24  | Q0+Q1 | BCE   | 1A best BCE / 1B-04 |
| 1B-25  | Q0+Q2 | B     | 1A best B / 1B-08   |
| 1B-26  | Q0+Q2 | BCE   | 1A best BCE / 1B-10 |
| 1B-27  | Q0+Q3 | E     | 1A best E / 1B-11   |
| 1B-28  | Q0+Q3 | BCE   | 1A best BCE / 1B-14 |
| 1B-29  | Q0+Q4 | B     | 1A best B / 1B-15   |
| 1B-30  | Q0+Q4 | E     | 1A best E / 1B-16   |
| 1B-31  | Q0+Q4 | BCE   | 1A best BCE / 1B-18 |

#### 1B-6: Combined Expansions (LLM-only and Q0-anchored)

> **Rationale:** This section tests multi-expansion combinations. `QALL` concatenates all 4 LLM expansions without Q0. `Q0+QALL` adds Q0 on top. By comparing them, we measure whether preserving the original query signal helps or hurts when many LLM terms are already present. BCE is used for all to control for model variation; B and E are added for Q0+QALL to check single-model behavior.

| Run ID | Query   | Model | Compare Against                    |
| ------ | ------- | ----- | ---------------------------------- |
| 1B-32  | QALL    | BCE   | Best of 1B-04, 1B-10, 1B-14, 1B-18 |
| 1B-33  | Q0+QALL | BCE   | 1B-32                              |
| 1B-34  | Q0+QALL | B     | Best B from 1B-5                   |
| 1B-35  | Q0+QALL | E     | Best E from 1B-5                   |

#### 1B-7: Query-Index Interactions

> **Rationale:** 1A found the best index config using Q0 only. But richer index text (F4/F5 with scope notes and parent SNCs) may interact differently with LLM-augmented queries. This section tests Q0+QALL (the most comprehensive query) against all 5 index configs to detect interactions missed by 1A.

| Run ID | Query   | Model | Index Config | Compare Against |
| ------ | ------- | ----- | ------------ | --------------- |
| 1B-36  | Q0+QALL | BCE   | F1           | 1B-33           |
| 1B-37  | Q0+QALL | BCE   | F2           | 1B-33           |
| 1B-38  | Q0+QALL | BCE   | F3           | 1B-33           |
| 1B-39  | Q0+QALL | BCE   | F4           | 1B-33           |
| 1B-40  | Q0+QALL | BCE   | F5           | 1B-33           |

**Total 1B: 40 runs**

---

### Experiment Set 1B-R: Revised Concept-Focused Queries (NEW)

**Purpose:** Test whether concept-focused query expansions outperform entity/event-focused expansions for matching against folder labels.
**Index:** F_best (best configuration per model from Set 1A)
**Evaluation:** nDCG@5 over 45 topics, single run

> **Design Rationale:**
> Each revised query is compared against its original counterpart (same model, same role) AND the Q0 baseline. This allows measuring: (a) does concept-focusing improve over entity-focusing? (b) does any expansion improve over Q0?

#### 1BR-1: Individual Revised Expansions

| Run ID | Query | Model | Compare Against | Tests                                     |
| ------ | ----- | ----- | --------------- | ----------------------------------------- |
| 1BR-01 | Q5    | B     | 1B-08 (Q2, B)   | Concept Bridge vs Keywords (B)            |
| 1BR-02 | Q5    | E     | 1B-11 (Q3, E)   | Concept Bridge vs EmbText (E)             |
| 1BR-03 | Q5    | BCE   | 1B-10 (Q2, BCE) | Concept Bridge vs Keywords (BCE)          |
| 1BR-04 | Q6    | B     | 1B-01 (Q1, B)   | SNC-Aware HyDE vs Original HyDE (B)       |
| 1BR-05 | Q6    | E     | 1B-02 (Q1, E)   | SNC-Aware HyDE vs Original HyDE (E)       |
| 1BR-06 | Q6    | BCE   | 1B-04 (Q1, BCE) | SNC-Aware HyDE vs Original HyDE (BCE)     |
| 1BR-07 | Q7    | E     | 1B-11 (Q3, E)   | Thematic vs Event-focused paragraph (E)   |
| 1BR-08 | Q7    | BCE   | 1B-14 (Q3, BCE) | Thematic vs Event-focused paragraph (BCE) |
| 1BR-09 | Q2R   | B     | 1B-08 (Q2, B)   | Term-focused vs Entity-focused KW (B)     |
| 1BR-10 | Q2R   | BCE   | 1B-10 (Q2, BCE) | Term-focused vs Entity-focused KW (BCE)   |

#### 1BR-2: Q0-Augmented Revised Expansions

| Run ID | Query | Model | Compare Against    | Tests                                      |
| ------ | ----- | ----- | ------------------ | ------------------------------------------ |
| 1BR-11 | Q0+Q5 | B     | 1B-25 (Q0+Q2, B)   | Concept augmented vs KW augmented (B)      |
| 1BR-12 | Q0+Q5 | E     | 1B-27 (Q0+Q3, E)   | Concept augmented vs Emb augmented (E)     |
| 1BR-13 | Q0+Q5 | BCE   | 1B-26 (Q0+Q2, BCE) | Concept augmented vs KW augmented (BCE)    |
| 1BR-14 | Q0+Q6 | B     | 1B-22 (Q0+Q1, B)   | SNC HyDE augmented vs HyDE augmented (B)   |
| 1BR-15 | Q0+Q6 | BCE   | 1B-24 (Q0+Q1, BCE) | SNC HyDE augmented vs HyDE augmented (BCE) |

#### 1BR-3: Combined Revised Expansions

| Run ID | Query     | Model | Compare Against      | Tests                               |
| ------ | --------- | ----- | -------------------- | ----------------------------------- |
| 1BR-16 | Q0+Q5+Q6  | BCE   | 1BR-13, 1BR-15       | Dual concept bridge (BCE)           |
| 1BR-17 | Q0+QALL-R | BCE   | 1B-33 (Q0+QALL, BCE) | Full revised vs full original (BCE) |
| 1BR-18 | Q0+QALL-R | B     | 1B-34 (Q0+QALL, B)   | Full revised vs full original (B)   |
| 1BR-19 | Q0+QALL-R | E     | 1B-35 (Q0+QALL, E)   | Full revised vs full original (E)   |

**Total 1B-R: 19 runs**

---

### Phase 1 Decisions (fill after running)

| Decision                                 | Value |
| ---------------------------------------- | ----- |
| D1 — F_best (overall)                   | ___   |
| D1a — F_best for B                      | ___   |
| D1b — F_best for E                      | ___   |
| D1c — F_best for C                      | ___   |
| D2 — Q_best_B (original)                | ___   |
| D3 — Q_best_E (original)                | ___   |
| D4 — Q_best_C (original)                | ___   |
| D5 — Q_best_BCE (original)              | ___   |
| D6 — Best overall (Q×F) (original)     | ___   |
| D20 — Q_best_B (revised or original?)   | ___   |
| D21 — Q_best_E (revised or original?)   | ___   |
| D22 — Q_best_BCE (revised or original?) | ___   |
| D23 — Revised > Original? (per model)   | ___   |

> These values are carried into ALL Phase 2 experiments. Q_best_R refers to the best revised query for each model (or the original if revised didn't improve).

**Phase 1 Total: 87 runs** (28 1A + 40 1B + 19 1B-R)

---

## Phase 2: Folder Label Augmentation

### Overview

Phase 2 replaces the original folder labels with LLM-generated augmented labels. The AllF pipeline then searches these enriched representations instead of the original labels.

**Key difference from Phase 1:** Augmented labels for folders *with* training documents depend on which documents are in the ECF. Therefore, all Phase 2 experiments are evaluated over 5 Uniform ECFs (mean ± 95% CI).

**Key revision:** Phase 2 now uses **SNC-adaptive prompts** that distinguish between specific and generic SNC folders.

### ECF-Dependency and Caching Logic

For any given ECF:

- Folder has training docs → use document-aware augmentation prompt (content varies per ECF)
- Folder has NO training docs → use pure metadata/knowledge prompt (content is ECF-independent)

**Optimization:** Cache by `(folder_id, frozenset(training_doc_ids))`. The same document set appearing in multiple ECFs for the same folder reuses the cached output. Folders without docs are generated once and shared across all ECFs.

**Pilot strategy:** Run the full augmentation pipeline on 1 ECF first. Review outputs before committing to all 5.

### Augmented Label Index Format

After generation, each folder's entry in the retrieval index will be:

```
{original_folder_label} | {expanded_snc} | {scope_note_if_available} | {DESCRIPTION} | {KEYWORDS}
```

The original label is preserved as a prefix (exact-match fallback) while the enriched content follows.
Scope note is included when available (37.7% of folders — 504/1,336). For the remainder, this field is omitted.
For Unknown-SNC folders, `expanded_snc` is derived from the folder label (see D18).

---

### Strategy 2A: Base Augmentation — Original (PRESERVED FOR COMPARISON)

#### Why

The LLM acts as an informed archivist. For folders with digitized documents, it grounds the augmented label in actual OCR evidence — extracting specific people, events, and dates. For empty folders, it uses the SNC classification and its historical knowledge of the 1960s-1970s Brazil-U.S. context.

#### Document Content to Use

Use `title + summary` per document (GPT-4o summaries already exist in the collection).

**Document selection strategy (data-driven — addresses POL dominance: 353 folders, 11,432 docs):**

- If a folder has ≤5 training documents: use all of them
- If a folder has >5 training documents: select **2 random documents from each of the
  folder's training docs** plus the top-3 by BM25 score against the folder label (capped at 5 total)
- **For SNC-level homophily (Phase 2B):** when an SNC has many folders with docs (e.g., POL sub-codes),
  select at most **2 random documents from each folder** within that SNC (capped at D8 total).
  This ensures diversity across folders rather than depth within one folder.

Fill decision D13 after reviewing pilot outputs.

#### PROMPT P2A-DOC (Folders WITH Training Documents — Original)

> **Implementation Notes (Phase 0 data-driven):**
>
> - If `snc == "Unknown"`: set SNC Code to `"Unclassified"`, derive SNC Meaning from `folder_label`
> - If `label_parent_expanded` is empty or equals `snc`: use `main_title` as Broader Category
> - If `scope_note` is empty: use `"No scope note available for this SNC code."`
> - If `end_date == "Unknown"`: use `"{start_date} (end date not recorded)"`

```
You are an expert archivist and historian specializing in U.S.-Brazil diplomatic relations during the 1960s and 1970s. Write enriched metadata for the archival folder below.

FOLDER INFORMATION:
- Folder Label: {folder_label}
- SNC Code: {snc} (or "Unclassified" if Unknown)
- SNC Meaning: {expanded_snc} (or derived from folder label if SNC is Unknown)
- Broader Category: {parent_expanded_snc} (or inferred from folder label if empty)
- Scope Note: {scope_note if available, otherwise "No scope note available for this SNC code."}
- Date Range: {start_date}{" to " + end_date if end_date != "Unknown" else " (end date not recorded)"}
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

#### PROMPT P2A-NODOC (Folders WITHOUT Training Documents — Original)

> **Same implementation notes as P2A-DOC apply** for Unknown SNC, missing dates, and scope notes.

```
You are an expert archivist and historian specializing in U.S.-Brazil diplomatic relations during the 1960s and 1970s. Write enriched metadata for an archival folder that has not yet been digitized.

FOLDER INFORMATION:
- Folder Label: {folder_label}
- SNC Code: {snc} (or "Unclassified" if Unknown)
- SNC Meaning: {expanded_snc} (or derived from folder label if SNC is Unknown)
- Broader Category: {parent_expanded_snc} (or inferred from folder label if empty)
- Scope Note: {scope_note if available, otherwise "No scope note available for this SNC code."}
- Date Range: {start_date}{" to " + end_date if end_date != "Unknown" else " (end date not recorded)"}
- Record Group: {record_group}

HISTORICAL CONTEXT:
This folder is from U.S. State Department records on Brazil, starting from {start_date}.
{"The period covers " + start_date + "–" + end_date + "." if end_date != "Unknown" else "The end date is not recorded, but based on the classification and folder context, content likely extends through the late 1960s or early 1970s."}
The broader historical period includes: the April 1964 military coup and subsequent military government, Cold War dynamics in Latin America, significant Brazilian economic development, and evolving U.S.-Brazil bilateral relations across security, trade, and diplomacy.

Think through these steps before writing:
Step 1 — SNC INTERPRETATION: What specific type of content does this SNC code cover? What would a State Department officer file under this label?
Step 2 — HISTORICAL EVENTS: What specific events, actors, and developments occurred in Brazil and in U.S.-Brazil relations relevant to this topic during the folder's time period?
Step 3 — DOCUMENT INFERENCE: What types of documents (cables, memos, intelligence reports, diplomatic notes) would likely populate this folder?

Then output the following two sections:

DESCRIPTION: A paragraph of 100-150 words starting exactly with "This folder likely contains" describing the probable historical content, likely key actors and institutions, and relevant events. Name known historical figures, institutions, and events from this era. Do not invent document contents — draw only from established historical knowledge.

KEYWORDS: A comma-separated list of 10-15 search terms including proper nouns, event names, institutions, and key concepts most likely to represent this folder.

Output only the DESCRIPTION and KEYWORDS sections. Do not output your step-by-step reasoning.
```

#### Experiment Set 2A (Original — PRESERVED)

*Each run is averaged over 5 Uniform ECFs.*

| Run ID                                                            | Augmentation | Query      | Model | Mask | Compare Against           |
| ----------------------------------------------------------------- | ------------ | ---------- | ----- | ---- | ------------------------- |
| 2A-01                                                             | Aug-Base     | Q0         | B     | Uni  | 1A best B (AllF baseline) |
| 2A-02                                                             | Aug-Base     | Q0         | E     | Uni  | 1A best E                 |
| 2A-03                                                             | Aug-Base     | Q0         | C     | Uni  | 1A best C                 |
| 2A-04                                                             | Aug-Base     | Q0         | BCE   | Uni  | 1A best BCE               |
| 2A-05                                                             | Aug-Base     | Q0         | BC    | Uni  | 1A best BC                |
| 2A-06                                                             | Aug-Base     | Q0         | BE    | Uni  | 1A best BE                |
| 2A-07                                                             | Aug-Base     | Q0         | CE    | Uni  | 1A best CE                |
| 2A-08                                                             | Aug-Base     | Q_best_B   | B     | Uni  | 2A-01                     |
| 2A-09                                                             | Aug-Base     | Q_best_E   | E     | Uni  | 2A-02                     |
| 2A-10                                                             | Aug-Base     | Q_best_C   | C     | Uni  | 2A-03                     |
| 2A-11                                                             | Aug-Base     | Q_best_BCE | BCE   | Uni  | 2A-04                     |
| 2A-12                                                             | Aug-Base     | Q_best_BCE | BC    | Uni  | 2A-05                     |
| 2A-13                                                             | Aug-Base     | Q_best_BCE | BE    | Uni  | 2A-06                     |
| 2A-14                                                             | Aug-Base     | Q_best_BCE | CE    | Uni  | 2A-07                     |
| **Total 2A: 14 runs (× 5 ECFs = 70 retrieval executions)** |              |            |       |      |                           |

---

### Strategy 2A-R: SNC-Adaptive Augmentation (NEW — REVISED)

#### Why

> **The single most important change in the revised experiments.** The original P2A prompts treat all SNCs identically: whether a folder is labeled "AGR Agriculture 1964" (specific) or "POL 15-1 Kennedy 1964" (generic), the same prompt asks for entities, events, and key figures. Data analysis reveals this is fundamentally mismatched:
>
> - **Specific SNCs** (AGR, DEF, LAB, SCI...): The label tells you the theme. The expansion needs *concept breadth* — related terms a user might search for. Documents are supplementary.
> - **Generic SNCs** (POL, E, SOC, ORG...): The label is too broad. The expansion needs *document-derived specificity*. Documents are the primary signal.

#### PROMPT P2AR-SPECIFIC-DOC (Specific SNC, WITH Documents)

```
You are an expert archivist specializing in U.S. State Department records on Brazil (1960s-1970s). Enrich the folder below by adding RELATED CONCEPTS AND TERMS that would help researchers find this folder.

FOLDER INFORMATION:
- Folder Label: {folder_label}
- SNC Category: {expanded_snc}
- Broader Category: {parent_expanded_snc}
- Scope Note: {scope_note}
- Date Range: {date_range_formatted}

DOCUMENTS IN THIS FOLDER:
{doc_list}
[Format: [Doc N] Title: ... | Summary: ...]

The SNC category "{expanded_snc}" is thematically clear. Your task is to:
1. Extract the KEY TERMS AND CONCEPTS from the documents (not people or dates — focus on topics, commodities, policies, programs, and domain terms)
2. Add RELATED CONCEPTS that a researcher might search for when looking for content in this thematic area (e.g., if the folder is about agriculture, add terms like "coffee, crops, agricultural policy, land reform, commodity exports")
3. Ground all terms in the Brazilian context of the 1960s-1970s

Output two sections:

DESCRIPTION: A paragraph of 80-120 words starting with "This folder contains" that describes the thematic content using the terms and concepts from the documents. Focus on WHAT TOPICS are covered, not WHO is mentioned. Include related terms that broaden findability.

KEYWORDS: A comma-separated list of 12-18 terms organized as:
- 3-4 broad category terms (matching SNC-level vocabulary)
- 4-6 specific concepts from the documents
- 4-6 related terms a researcher might use to search for this content
- 2-3 Portuguese terms if relevant

Output only DESCRIPTION and KEYWORDS. No reasoning steps.
```

---

#### PROMPT P2AR-SPECIFIC-NODOC (Specific SNC, WITHOUT Documents)

```
You are an expert archivist specializing in U.S. State Department records on Brazil (1960s-1970s). Enrich the folder below by adding RELATED CONCEPTS AND TERMS.

FOLDER INFORMATION:
- Folder Label: {folder_label}
- SNC Category: {expanded_snc}
- Broader Category: {parent_expanded_snc}
- Scope Note: {scope_note}
- Date Range: {date_range_formatted}

The SNC category "{expanded_snc}" is thematically clear. No documents are available for this folder. Your task is to:
1. Infer what TOPICS AND CONCEPTS would logically be filed under this SNC during the folder's time period in the Brazil context
2. Add RELATED TERMS that a researcher might search for (e.g., if the SNC is about agriculture, add "coffee, sugar, cocoa, crop yields, land reform, agrarian policy")
3. Think about what specific aspects of {expanded_snc} were relevant in Brazil during {date_range_formatted}

Output two sections:

DESCRIPTION: A paragraph of 80-120 words starting with "This folder likely contains" describing the probable thematic content. Focus on concepts and terms, not specific events or people. Include related terms that broaden findability.

KEYWORDS: A comma-separated list of 12-18 terms as described above.

Output only DESCRIPTION and KEYWORDS. No reasoning steps.
```

---

#### PROMPT P2AR-GENERIC-DOC (Generic SNC, WITH Documents — Document-Driven)

```
You are an expert archivist specializing in U.S. State Department records on Brazil (1960s-1970s). This folder has a GENERIC classification that doesn't reveal its specific content. The documents are the primary signal for what this folder actually contains.

FOLDER INFORMATION:
- Folder Label: {folder_label}
- SNC Category: {expanded_snc}
- Broader Category: {parent_expanded_snc}
- Scope Note: {scope_note}
- Date Range: {date_range_formatted}

DOCUMENTS IN THIS FOLDER (these are your PRIMARY source of context):
{doc_list}
[Format: [Doc N] Title: ... | Summary: ...]

The SNC category "{expanded_snc}" is BROAD and could cover many sub-topics. The documents above are your best evidence of what this specific folder contains. Your task is to:
1. CAREFULLY read each document title and summary to identify the MAIN THEMES discussed
2. Determine what specific sub-topic of "{expanded_snc}" this folder focuses on
3. Generate terms and concepts that capture the folder's actual content (as revealed by the documents) — not just the generic SNC label
4. Add related concepts that would help a researcher find this folder when searching for the specific topics discussed in these documents

Output two sections:

DESCRIPTION: A paragraph of 100-150 words starting with "This folder contains" that describes the SPECIFIC thematic content revealed by the documents. Name the actual topics discussed (e.g., "military governance", "economic stabilization", "student protests") — not generic phrases like "political affairs". Include related concepts that broaden the match beyond what's literally in the documents.

KEYWORDS: A comma-separated list of 15-20 terms:
- 2-3 broad category terms
- 6-8 specific terms extracted from document themes
- 4-6 related concepts a researcher would associate with these themes
- 2-3 Portuguese terms if relevant

Output only DESCRIPTION and KEYWORDS. No reasoning steps.
```

---

#### PROMPT P2AR-GENERIC-NODOC (Generic SNC, WITHOUT Documents — Hardest Case)

```
You are an expert archivist specializing in U.S. State Department records on Brazil (1960s-1970s). This folder has a GENERIC classification and NO digitized documents.

FOLDER INFORMATION:
- Folder Label: {folder_label}
- SNC Category: {expanded_snc}
- Broader Category: {parent_expanded_snc}
- Scope Note: {scope_note}
- Date Range: {date_range_formatted}

The SNC category "{expanded_snc}" is BROAD. Without documents, you must rely on:
1. The SPECIFIC FOLDER LABEL "{folder_label}" — look for any geographic, temporal, or topical qualifiers that narrow the scope
2. The DATE RANGE — what specific aspects of "{expanded_snc}" were most active in Brazil during {date_range_formatted}?
3. The SCOPE NOTE (if available) — what does the classification system say about this code?

Your task: Generate the MOST LIKELY thematic content for this folder. Since the SNC is generic, be SPECIFIC about sub-topics rather than repeating the broad category.

Output two sections:

DESCRIPTION: A paragraph of 80-120 words starting with "This folder likely contains" that describes probable specific sub-topics of "{expanded_snc}" relevant to the folder's label and time period. Be specific — don't just repeat the SNC category name.

KEYWORDS: A comma-separated list of 12-18 terms covering:
- The broad category + 3-4 likely specific sub-topics
- 5-8 related concepts and terms a researcher might use
- 2-3 Portuguese terms if relevant

Output only DESCRIPTION and KEYWORDS. No reasoning steps.
```

---

#### Experiment Set 2A-R

*Each run is averaged over 5 Uniform ECFs.*

| Run ID | Augmentation | SNC Handling                  | Query    | Model | Mask | Compare Against   |
| ------ | ------------ | ----------------------------- | -------- | ----- | ---- | ----------------- |
| 2AR-01 | SNC-Adaptive | Split specific/generic        | Q0       | BCE   | Uni  | 2A-04 (original)  |
| 2AR-02 | SNC-Adaptive | Split specific/generic        | Q0       | B     | Uni  | 2A-01 (original)  |
| 2AR-03 | SNC-Adaptive | Split specific/generic        | Q0       | E     | Uni  | 2A-02 (original)  |
| 2AR-04 | SNC-Adaptive | Split specific/generic        | Q_best_R | BCE   | Uni  | 2AR-01            |
| 2AR-05 | SNC-Adaptive | Split specific/generic        | Q0+Q5    | BCE   | Uni  | 2AR-01            |
| 2AR-06 | SNC-Adaptive | Uniform (all specific prompt) | Q0       | BCE   | Uni  | 2AR-01 (ablation) |
| 2AR-07 | SNC-Adaptive | Uniform (all generic prompt)  | Q0       | BCE   | Uni  | 2AR-01 (ablation) |

**Total 2A-R: 7 runs (× 5 ECFs = 35 retrieval executions)**

> **Ablation runs 2AR-06 and 2AR-07:** These test whether the SNC-adaptive split is genuinely necessary by applying each strategy uniformly to all folders. If split ≈ uniform-specific, concept-broadening alone is sufficient. If split ≈ uniform-generic, document-context alone is sufficient. If split > both, the adaptive approach adds value.

---

### Strategy 2B: Homophily Augmentation — Original (PRESERVED FOR COMPARISON)

#### Why

The archival principle of *original order* implies that folders filed under the same SNC code share thematic content. When an empty folder shares an SNC code with folders that DO have training documents, those documents serve as thematic proxies.

#### Prerequisite

**Requires completed Phase 0, Sections 0.2 and 0.3.** ✔️ Phase 0 completed — decisions pre-filled below.

#### Decisions (Pre-filled from Phase 0 Data Analysis)

- D6 — Similarity criterion: **☑ Cascading hierarchy** (recommended based on Phase 0 data):
  1. First try **exact SNC match** → if ≥ D7 docs found, use them
  2. If insufficient, expand to **2-level SNC prefix** → if ≥ D7 docs found, use them
  3. If still insufficient, expand to **parent SNC** → if ≥ D7 docs found, use them
  4. If still insufficient, fall back to **P2A-NODOC** (pure knowledge prompt)
     Record which level was used for each folder (for per-level analysis).
- D7 — Minimum neighbor docs threshold: ___ (suggested: 3)
- D8 — Max neighbor docs in prompt: ___ (suggested: 5)
- D9 — Proceed with Homophily? **☑ Yes** — with cascading hierarchy, **73.8%** of empty folders have ≥3 parent-level neighbor docs, well above the 20% viability threshold.
- D19 — Max docs per folder in neighbor sampling: ___ (suggested: 2 — ensures diversity for large SNCs like POL)

> **Phase 0 Evidence:**
>
> | Match Level | ≥1 neighbor doc | ≥3 neighbor docs |
> | ----------- | ---------------- | ----------------- |
> | Exact SNC   | 541/774 (69.9%)  | 333/774 (43.0%)   |
> | 2-Level SNC | 595/774 (76.9%)  | 426/774 (55.0%)   |
> | Parent SNC  | 696/774 (89.9%)  | 571/774 (73.8%)   |

#### PROMPT P2B-HOMO (Original — Folders WITHOUT Documents, with Neighbor Context)

```
You are an expert archivist and historian specializing in U.S.-Brazil diplomatic relations during the 1960s and 1970s. Write enriched metadata for an archival folder that has not been digitized.

TARGET FOLDER (not yet digitized):
- Folder Label: {folder_label}
- SNC Code: {snc} (or "Unclassified" if Unknown)
- SNC Meaning: {expanded_snc} (or derived from folder label if SNC is Unknown)
- Date Range: {start_date}{" to " + end_date if end_date != "Unknown" else " (end date not recorded)"}

CONTEXTUAL DOCUMENTS FROM RELATED FOLDERS:
The documents below come from OTHER folders with a related SNC classification (match level: {match_level} — e.g., exact "{snc}", 2-level "{snc_2level}", or parent "{parent}"). They do NOT belong to the target folder. Use them to understand what topics, actors, and events are typically filed under this classification — but do not assume their specific events apply to the target folder.

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

#### Experiment Set 2B (Original — PRESERVED)

| Run ID | Augmentation | Query      | Model | Mask | Compare Against                                |
| ------ | ------------ | ---------- | ----- | ---- | ---------------------------------------------- |
| 2B-01  | Aug-Homo     | Q0         | BCE   | Uni  | 2A-04 (same query, different empty-folder aug) |
| 2B-02  | Aug-Homo     | Q_best_BCE | BCE   | Uni  | 2B-01                                          |
| 2B-03  | Aug-Homo     | Q0         | B     | Uni  | 2A-01                                          |
| 2B-04  | Aug-Homo     | Q_best_B   | B     | Uni  | 2B-03                                          |
| 2B-05  | Aug-Homo     | Q0         | E     | Uni  | 2A-02                                          |
| 2B-06  | Aug-Homo     | Q_best_E   | E     | Uni  | 2B-05                                          |
| 2B-07  | Aug-Homo     | Q0         | C     | Uni  | 2A-03                                          |
| 2B-08  | Aug-Homo     | Q_best_C   | C     | Uni  | 2B-07                                          |

**Total 2B: 8 runs**

---

### Strategy 2B-R: SNC-Adaptive Homophily (NEW — REVISED)

#### Why

The original homophily treats all neighbor document context uniformly. The data-driven insight is that the *role* of neighbor documents differs by SNC specificity:

- **Specific SNC + Homophily:** Neighbor documents are *supplementary*. They add concrete examples (e.g., specific crops, specific labor actions) to an already clear thematic category. The expansion still focuses on concept breadth.
- **Generic SNC + Homophily:** Neighbor documents are *essential*. They're the only way to understand what sub-topics a generic classification like POL actually covers. The expansion must extract and project the thematic patterns.

#### PROMPT P2BR-HOMO-SPECIFIC (Specific SNC, Homophily)

```
You are an expert archivist specializing in U.S. State Department records on Brazil (1960s-1970s). Write enriched metadata for an archival folder using context from related folders.

TARGET FOLDER (not digitized):
- Folder Label: {folder_label}
- SNC Category: {expanded_snc}
- Date Range: {date_range_formatted}

DOCUMENTS FROM RELATED FOLDERS (same SNC classification, match level: {match_level}):
{neighbor_docs}
[Format: [Doc N] From: {neighbor_label} | Title: ... | Summary: ...]

The SNC "{expanded_snc}" is thematically SPECIFIC. The related documents provide EXAMPLES of concrete content filed under this category. Use them to:
1. Identify concrete terms and concepts that appear across the related folders
2. Add RELATED TERMS that broaden the match (e.g., if documents discuss "coffee exports", also add "agricultural commodities, crop production, trade balance")
3. Distinguish the TARGET folder from neighbors using its label and date range

Output DESCRIPTION (80-120 words, "This folder likely contains") and KEYWORDS (12-18 terms).
No reasoning steps.
```

---

#### PROMPT P2BR-HOMO-GENERIC (Generic SNC, Homophily — CRITICAL)

```
You are an expert archivist specializing in U.S. State Department records on Brazil (1960s-1970s). Write enriched metadata for a GENERICALLY CLASSIFIED folder using documents from related folders as your PRIMARY context source.

TARGET FOLDER (not digitized):
- Folder Label: {folder_label}
- SNC Category: {expanded_snc}
- Date Range: {date_range_formatted}

DOCUMENTS FROM RELATED FOLDERS (match level: {match_level}):
{neighbor_docs}
[Format: [Doc N] From: {neighbor_label} | Title: ... | Summary: ...]

The SNC "{expanded_snc}" is BROAD and generic. The documents above are your MOST IMPORTANT source for understanding what specific sub-topics exist under this classification. Your task:

1. READ CAREFULLY: What specific themes, topics, and concerns appear across these documents? Look for patterns — are they about governance, security, diplomacy, protests, economic policy?
2. INFER: Given the target folder's specific label "{folder_label}" and date range, which of these themes most likely applies?
3. EXPAND: Add related concepts that a researcher might search for when looking for content about these specific themes (not the broad SNC category)

Output DESCRIPTION (100-150 words, "This folder likely contains" — be SPECIFIC about sub-topics, not generic about the SNC) and KEYWORDS (15-20 terms, emphasizing document-derived themes and related concepts).
No reasoning steps.
```

---

#### Experiment Set 2B-R

| Run ID | Augmentation        | SNC Handling   | Query    | Model | Mask | Compare Against |
| ------ | ------------------- | -------------- | -------- | ----- | ---- | --------------- |
| 2BR-01 | SNC-Adaptive + Homo | Split-weighted | Q0       | BCE   | Uni  | 2AR-01, 2B-01   |
| 2BR-02 | SNC-Adaptive + Homo | Split-weighted | Q_best_R | BCE   | Uni  | 2BR-01          |
| 2BR-03 | SNC-Adaptive + Homo | Split-weighted | Q0       | B     | Uni  | 2AR-02          |
| 2BR-04 | SNC-Adaptive + Homo | Split-weighted | Q0       | E     | Uni  | 2AR-03          |

**Total 2B-R: 4 runs (× 5 ECFs = 20 retrieval executions)**

---

### Strategy 2C: TCDE Augmentation (UNCHANGED)

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

#### Experiment Set 2C (UNCHANGED)

| Run ID | Augmentation | Query      | Model | Mask | Compare Against                  |
| ------ | ------------ | ---------- | ----- | ---- | -------------------------------- |
| 2C-01  | Aug-TCDE     | Q0         | BCE   | Uni  | 2A-04 (base aug, same query)     |
| 2C-02  | Aug-TCDE     | Q_best_BCE | BCE   | Uni  | 2C-01                            |
| 2C-03  | Aug-TCDE     | Q-TCDE     | BCE   | Uni  | 2C-02 (TCDE folder + TCDE query) |
| 2C-04  | Aug-TCDE     | Q0         | B     | Uni  | 2A-01                            |
| 2C-05  | Aug-TCDE     | Q_best_B   | B     | Uni  | 2C-04                            |
| 2C-06  | Aug-TCDE     | Q0         | E     | Uni  | 2A-02                            |
| 2C-07  | Aug-TCDE     | Q_best_E   | E     | Uni  | 2C-06                            |
| 2C-08  | Aug-TCDE     | Q0         | C     | Uni  | 2A-03                            |
| 2C-09  | Aug-TCDE     | Q_best_C   | C     | Uni  | 2C-08                            |

**Total 2C: 9 runs**

---

### Strategy 2T: Topic-Aware Collection Modeling (NEW)

#### Motivation

Instead of treating each folder independently, model the collection as a set of *latent topics* and match queries to these discovered topics first, then use topic membership to adjust folder ranking.

> **CRITICAL CONSTRAINT:** This strategy does NOT use QRELs (relevance judgments) at any point. All topic discovery is unsupervised, derived purely from the collection metadata and documents.

#### Why This Approach

The 1,336 folders are organized by SNC codes, but the SNC hierarchy alone doesn't capture the full thematic diversity. For example, POL contains 561 folders spanning everything from elections to coups to diplomatic visits. Topic modeling discovers *latent sub-themes* within these broad categories that help distinguish which folders are relevant for which types of queries.

#### Two-Phase Approach

```
Phase A: Unsupervised Topic Discovery (from collection data only)
  → Discover K latent topics from folder labels + augmented descriptions + document summaries
  → Assign each folder to its most representative topic(s)

Phase B: Query-Topic Matching (at query time)
  → Classify an incoming query into discovered topics
  → Use topic membership as a prior to re-weight folder scores
```

#### Step 1: Build Folder Topic Representations

Construct a text representation for each folder using all available metadata (no QRELs):

```python
def build_folder_topic_text(folder_id: str, folder_meta: dict, 
                             items_meta: dict, ecf_docs: dict) -> str:
    """
    Build a rich text representation of a folder for topic modeling.
    Uses ONLY collection data, never QRELs.
    """
    parts = []
  
    # Folder metadata
    parts.append(folder_meta.get('main_title', ''))
    parts.append(folder_meta.get('label_parent_expanded', ''))
    scope = folder_meta.get('scope_truncated', '') or folder_meta.get('raw_scope', '')
    if scope:
        parts.append(scope)
  
    # Document content (if available in ECF)
    doc_ids = ecf_docs.get(folder_id, [])
    for doc_id in doc_ids[:5]:  # Cap at 5 docs
        doc = items_meta.get(doc_id, {})
        if doc.get('title'):
            parts.append(doc['title'])
        if doc.get('summary'):
            parts.append(doc['summary'])
  
    return ' '.join(filter(None, parts))
```

#### Step 2: Discover Latent Topics (Unsupervised)

Use BERTopic or LDA on the folder representations to discover K latent themes:

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

def discover_collection_topics(folder_texts: dict, n_topics: int = 15):
    """
    Unsupervised topic discovery from collection data.
    No QRELs involved — purely based on folder/document content.
    """
    folder_ids = list(folder_texts.keys())
    texts = [folder_texts[fid] for fid in folder_ids]
  
    # Use same embedding model as retrieval for consistency
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
  
    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=n_topics,
        min_topic_size=10,  # Minimum folders per topic
        verbose=True
    )
  
    topics, probs = topic_model.fit_transform(texts)
  
    # Build folder → topic mapping
    folder_topics = {fid: (topic, prob) 
                     for fid, topic, prob in zip(folder_ids, topics, probs)}
  
    # Get topic descriptions (top words per topic)
    topic_info = topic_model.get_topic_info()
  
    return folder_topics, topic_info, topic_model
```

#### Step 3: Generate Topic Summaries for Query Matching

Use the LLM to generate natural-language descriptions of each discovered topic:

```
PROMPT P2T-SUMMARIZE:

You are an expert on U.S. State Department archival classification. Below are the top 
keywords for a group of related archival folders from a collection on Brazil (1960s-1970s).

Topic Keywords: {top_words}
Example folder labels from this group:
{sample_folder_labels}

Write a 2-sentence description of what theme or subject area this group of folders covers.
Then list 5-10 research questions or search topics that a researcher might use to find 
folders in this group.

Format:
THEME: [2-sentence description]
QUERIES: [comma-separated list of likely search topics]
```

#### Step 4: Query-Topic Matching at Retrieval Time

For an incoming query, predict which discovered topics are most relevant:

```
PROMPT P2T-CLASSIFY:

You are an expert on U.S. State Department archival collections on Brazil (1960s-1970s). 
A researcher is searching the archive with the following topic:

Research Topic Title: {title}
Research Topic Description: {description}

Below are {K} thematic groups discovered in the archive. Each group contains folders 
sharing a common theme:

{for each topic:}
Group {i}: {topic_theme_description}
  Likely search topics: {topic_likely_queries}
  Number of folders: {folder_count}
{end for}

Which groups are MOST LIKELY to contain folders relevant to the researcher's topic?
Rate each group on a scale of 0-10 (0 = completely unrelated, 10 = highly relevant).
Only list groups with score ≥ 3.

Format: Group N: score
```

#### Step 5: Apply Topic Prior to Retrieval Scores

```python
def apply_topic_prior(base_scores: dict, query_topic_scores: dict, 
                       folder_topics: dict, alpha: float = 0.2) -> dict:
    """
    Re-weight folder scores by incorporating discovered topic relevance.
  
    base_scores: {folder_id: retrieval_score} from Phase 1/2 retrieval
    query_topic_scores: {topic_id: relevance_score} from P2T-CLASSIFY
    folder_topics: {folder_id: (topic_id, membership_prob)} from Step 2
    alpha: interpolation weight (0 = pure retrieval, 1 = pure topic prior)
    """
    reweighted = {}
  
    # Normalize query topic scores
    max_score = max(query_topic_scores.values()) if query_topic_scores else 1
  
    for folder_id, base_score in base_scores.items():
        folder_topic, membership_prob = folder_topics.get(folder_id, (-1, 0.0))
    
        # Topic prior: how relevant is this folder's topic to the query?
        topic_relevance = query_topic_scores.get(folder_topic, 0) / max_score
    
        # Weight by membership probability (how strongly this folder belongs to its topic)
        weighted_prior = topic_relevance * membership_prob
    
        reweighted[folder_id] = (1 - alpha) * base_score + alpha * weighted_prior
  
    return reweighted
```

#### Code to Implement

```
src/topic_modeling/
├── build_representations.py   # Step 1: Build folder topic texts
├── discover_topics.py         # Step 2: BERTopic/LDA unsupervised discovery
├── summarize_topics.py        # Step 3: LLM-generated topic summaries
├── classify_query.py          # Step 4: Query-topic matching
├── apply_prior.py             # Step 5: Score re-weighting
└── run_topic_pipeline.py      # End-to-end pipeline
```

#### Experiment Set 2T

> **Note:** Topic modeling uses the ECF documents for building folder representations. For fair comparison, topic models are rebuilt per ECF (different document availability changes folder representations). However, topic discovery is unsupervised — no QRELs used.

| Run ID | Strategy                       | Alpha | Query    | Model | Mask | Compare Against         |
| ------ | ------------------------------ | ----- | -------- | ----- | ---- | ----------------------- |
| 2T-01  | Topic-Prior                    | 0.1   | Q0       | BCE   | Uni  | 2AR-01 (no topic prior) |
| 2T-02  | Topic-Prior                    | 0.2   | Q0       | BCE   | Uni  | 2T-01                   |
| 2T-03  | Topic-Prior                    | 0.3   | Q0       | BCE   | Uni  | 2T-02                   |
| 2T-04  | Topic-Prior (best α)          | best  | Q_best_R | BCE   | Uni  | 2T-best                 |
| 2T-05  | Topic-Prior + SNC-Adaptive Aug | best  | Q0       | BCE   | Uni  | 2AR-01, 2T-03           |
| 2T-06  | Topic-Prior + SNC-Adaptive Aug | best  | Q_best_R | BCE   | Uni  | 2T-05                   |

**Total 2T: 6 runs (× 5 ECFs = 30 retrieval executions)**

---

### Cross-Strategy Comparison

After 2A, 2A-R, 2B, 2B-R, 2C, 2T, test whether combining strategies for different folder tiers outperforms any single strategy.

| Run ID | Augmentation                   | Query      | Model | Mask | Logic                                           |
| ------ | ------------------------------ | ---------- | ----- | ---- | ----------------------------------------------- |
| 2X-01  | Hybrid (2A-doc + 2C-nodoc)     | Q_best_BCE | BCE   | Uni  | Folders with docs → P2A-DOC; empty → P2C      |
| 2X-02  | Hybrid (2A-doc + 2B-nodoc)     | Q_best_BCE | BCE   | Uni  | Folders with docs → P2A-DOC; empty → P2B-HOMO |
| 2X-03  | Hybrid-R (2AR-doc + 2BR-nodoc) | Q_best_R   | BCE   | Uni  | Revised: SNC-adaptive + homo (NEW)              |
| 2X-04  | Best-of + Topic-Prior          | Q_best_R   | BCE   | Uni  | Best augmentation + topic prior (NEW)           |

**Total 2X: 4 runs**

### Phase 2 Decisions (fill after running)

| Decision                                       | Value                                             |
| ---------------------------------------------- | ------------------------------------------------- |
| D10 — Aug_best (best single strategy, Uni)    | ___                                               |
| D11 — Does Hybrid outperform single strategy? | ☐ Yes / ☐ No                                    |
| D12 — N topics for TCDE                       | ___ (5 recommended, adjust if pilot shows issues) |
| D13 — Document content for prompts            | ☐ title+summary / ☐ title+OCR-p1 / ☐ full-OCR  |
| D14 — LLM for augmentation                    | ___                                               |
| D24 — SNC-Adaptive > Original prompts?        | ☐ Yes / ☐ No                                    |
| D25 — Topic Prior improves ranking?           | ☐ Yes / ☐ No                                    |
| D26 — Best alpha for Topic Prior              | ___                                               |
| D27 — Best K (number of topics)               | ___                                               |

---

## Phase 3: Re-ranking (REVISED — with Intra-SNC Precision Strategies)

### Objective

Re-rank the top-100 folders from the best Phase 2 run. Phase 2 gives an initial ranking based on augmented folder labels. Phase 3 adds a second signal: actual training document scores (3A), LLM reasoning (3B), or **document-evidence precision scoring to disambiguate within SNC groups (3C/3D)**.

### Why Re-ranking Helps

Phase 2 augmented labels are LLM-inferred — useful but imperfect. For folders that DO have training documents, scoring those documents directly against the query gives harder evidence of relevance. For folders without documents, LLM reasoning about which candidate is more likely to contain relevant material applies global context that the vector similarity alone cannot.

> **NEW Problem — Intra-SNC Disambiguation:**
>
> With concept-focused expansion (Phase 1B-R + 2A-R), queries like "coffee" will successfully match the AGR SNC group. But this creates a *new* problem: **all 82 AGR folders now have similar retrieval scores** because their augmented labels all contain agriculture-related terms. The retrieval system correctly identifies the right *category* but can't distinguish *which* folders within that category are actually about coffee vs. sugar vs. livestock.
>
> Strategies 3C and 3D specifically target this intra-SNC precision problem by using document-level evidence and targeted LLM reasoning to disambiguate folders that share the same broad SNC classification.

### Strategy 3A: Traditional Document-Based Re-ranking

#### Scoring Rule

```
if folder has training docs:
    score = max(BM25F_score(query, doc) for doc in training_docs)
else:
    score = initial_phase2_score * decay_factor  (e.g., 0.9)
```

Then re-sort top-100 by new scores. Apply Reciprocal Rank Fusion if multiple models are used for document scoring.

#### Experiment Set 3A

| Run ID | Base Phase 2 Run | Re-rank Model | k   | Mask | Compare Against     |
| ------ | ---------------- | ------------- | --- | ---- | ------------------- |
| 3A-01  | Best 2A (Uni)    | B             | 100 | Uni  | Best 2A run         |
| 3A-02  | Best 2A (Uni)    | E             | 100 | Uni  | 3A-01               |
| 3A-03  | Best 2A (Uni)    | C             | 100 | Uni  | 3A-01               |
| 3A-04  | Best 2A (Uni)    | BCE           | 100 | Uni  | 3A-01               |
| 3A-05  | Best 2B (Uni)    | BCE           | 100 | Uni  | Best 2B run         |
| 3A-06  | Best 2C (Uni)    | BCE           | 100 | Uni  | Best 2C run         |
| 3A-07  | Best 2A-R (Uni)  | BCE           | 100 | Uni  | Best 2A-R run (NEW) |
| 3A-08  | Best 2X (Uni)    | BCE           | 100 | Uni  | Best 2X run (NEW)   |

**Total 3A: 8 runs**

---

### Strategy 3B: LLM-Based Re-ranking

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

| Run ID | Base Phase 2 Run   | LLM   | k  | Mask | Compare Against    |
| ------ | ------------------ | ----- | -- | ---- | ------------------ |
| 3B-01  | Best 2A (Uni)      | {D15} | 20 | Uni  | 3A-04, Best 2A Uni |
| 3B-02  | Best 2C (Uni)      | {D15} | 20 | Uni  | 3A-06, Best 2C Uni |
| 3B-03  | Best Revised (Uni) | {D15} | 20 | Uni  | 3A-07 (NEW)        |

**Total 3B: 3 runs**

---

### Strategy 3C: Document-Evidence Precision Re-ranking (NEW)

#### Why

This is the critical strategy for the **intra-SNC disambiguation problem**. After concept-focused expansion retrieves all folders from the correct SNC group (e.g., all AGR folders for a "coffee" query), the initial retrieval scores are nearly identical because all folders share similar augmented labels. The key differentiator is the **actual document content**: folders whose training documents mention coffee, coffee exports, or coffee production should rank above folders whose documents discuss livestock or sugar.

#### The Two-Signal Architecture

```
Stage 1 (Phase 1/2 Retrieval):  Query → broad concept match → top-K folders
                                  "coffee" → all AGR, some INCO, some E folders
                              
Stage 2 (3C Precision Rerank):  Original query terms → document evidence → precise ranking
                                  "coffee" → check docs: which AGR folders have coffee?
```

> **Critical Design Choice:** The precision re-ranking uses the **ORIGINAL query terms (Q0)**, not the expanded query. The expanded query (Q5/Q6/Q7) is optimized for *broad category matching* — it intentionally adds terms like "agriculture, crops, commodities" that would match ALL AGR folders equally. The original query ("coffee") is the *precision signal* that disambiguates within the category.

#### Scoring Rule

```python
def precision_rerank(query_original: str, top_k_folders: list, 
                     ecf_docs: dict, items_meta: dict,
                     folders_meta: dict, model, 
                     alpha: float = 0.4, decay: float = 0.7) -> list:
    """
    Re-rank top-K folders using document evidence scored against
    the ORIGINAL query (not expanded).
  
    alpha: weight of document evidence vs initial score
    decay: penalty for folders without any document evidence
    """
    reranked = []
  
    for folder_id, init_score in top_k_folders:
        doc_ids = ecf_docs.get(folder_id, [])
    
        if doc_ids:
            # CASE 1: Folder HAS training documents
            # Score each doc against the ORIGINAL query (precision signal)
            doc_scores = []
            for doc_id in doc_ids:
                doc = items_meta.get(doc_id, {})
                doc_text = f"{doc.get('title', '')} {doc.get('summary', '')}"
                score = model.score(query_original, doc_text)
                doc_scores.append(score)
        
            # Take max (best-matching document) and mean (overall relevance)
            max_doc_score = max(doc_scores)
            mean_doc_score = sum(doc_scores) / len(doc_scores)
            doc_evidence = 0.7 * max_doc_score + 0.3 * mean_doc_score
        
            final_score = (1 - alpha) * init_score + alpha * doc_evidence
        
        else:
            # CASE 2: Folder has NO training documents
            # Check same-SNC neighbors for borrowed evidence
            snc = folders_meta[folder_id].get('snc', 'Unknown')
            neighbor_docs = find_same_snc_docs(folder_id, snc, ecf_docs, 
                                               folders_meta, max_docs=5)
        
            if neighbor_docs:
                # CASE 2a: Borrow evidence from SNC neighbors
                neighbor_scores = []
                for doc_id in neighbor_docs:
                    doc = items_meta.get(doc_id, {})
                    doc_text = f"{doc.get('title', '')} {doc.get('summary', '')}"
                    score = model.score(query_original, doc_text)
                    neighbor_scores.append(score)
            
                # Borrowed evidence weighted lower (it's indirect)
                borrowed_evidence = max(neighbor_scores) * 0.6
                final_score = (1 - alpha) * init_score + alpha * borrowed_evidence
            else:
                # CASE 2b: No evidence at all — apply decay
                final_score = init_score * decay
    
        reranked.append((folder_id, final_score))
  
    # Re-sort by final score
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked
```

#### Why Use Q0 (Not Expanded Query) for Scoring

| Query                      | Purpose                            | Example for "coffee" topic                                                      |
| -------------------------- | ---------------------------------- | ------------------------------------------------------------------------------- |
| **Expanded (Q0+Q5)** | Stage 1: Find the right SNC group  | "coffee agriculture crops commodities exports trade" → matches ALL AGR folders |
| **Original (Q0)**    | Stage 2: Disambiguate within group | "coffee" → only matches docs that actually mention coffee                      |

Using the expanded query for re-ranking would defeat the purpose — it would still score all AGR folders similarly. The original query is the precision knife.

#### Code to Implement

```
src/reranking/
├── rerank_traditional.py      # 3A: existing
├── rerank_llm.py              # 3B: existing
├── rerank_precision.py        # 3C: NEW — document-evidence precision
│   ├── score_folder_docs(query_original, folder_id, ecf_docs, items_meta, model)
│   │     → float (document evidence score)
│   ├── find_same_snc_docs(folder_id, snc, ecf_docs, folders_meta, max_docs)
│   │     → [doc_id, ...] (borrowed docs from same-SNC neighbors)
│   ├── precision_rerank(query_original, top_k, ecf_docs, items_meta, 
│   │                     folders_meta, model, alpha, decay)
│   │     → reranked [(folder_id, score), ...]
│   └── run_precision_reranking(base_run, ecf_path, model, alpha, decay)
│         → saves reranked run file
└── rerank_llm_precision.py    # 3D: NEW — LLM intra-group disambiguation
```

#### Experiment Set 3C

| Run ID | Base Phase 2 Run | Re-rank Query | Re-rank Model | k   | Alpha | Mask | Compare Against            |
| ------ | ---------------- | ------------- | ------------- | --- | ----- | ---- | -------------------------- |
| 3C-01  | Best 2A-R (Uni)  | Q0 (original) | B             | 100 | 0.3   | Uni  | 3A-07 (traditional rerank) |
| 3C-02  | Best 2A-R (Uni)  | Q0 (original) | E             | 100 | 0.3   | Uni  | 3C-01                      |
| 3C-03  | Best 2A-R (Uni)  | Q0 (original) | BCE           | 100 | 0.3   | Uni  | 3C-01                      |
| 3C-04  | Best 2A-R (Uni)  | Q0 (original) | BCE           | 100 | 0.5   | Uni  | 3C-03 (alpha ablation)     |
| 3C-05  | Best 2A-R (Uni)  | Q0 (original) | BCE           | 100 | 0.7   | Uni  | 3C-03 (alpha ablation)     |
| 3C-06  | Best 2X (Uni)    | Q0 (original) | BCE           | 100 | best  | Uni  | 3A-08                      |
| 3C-07  | Best 2BR (Uni)   | Q0 (original) | BCE           | 100 | best  | Uni  | Best 2BR run               |

**Total 3C: 7 runs**

---

### Strategy 3D: LLM Intra-Group Disambiguation (NEW)

#### Why

Strategy 3C uses document evidence mechanistically (BM25/embedding scores). Strategy 3D uses the LLM to reason about which folders within a *cluster of similar candidates* are most likely to contain the specific content the query asks for. This is particularly useful when:

1. **Multiple folders from the same SNC lack documents** — the LLM can use its knowledge to reason about which specific folder label (with its date and geographic qualifiers) is most likely relevant
2. **The query is very specific** (e.g., "Alagoas flood 1969") but many POL folders match broadly — the LLM can use the date range and label to make precise judgments

#### Approach: Cluster-Then-Rank

```
1. Take top-K from retrieval (e.g., K=50)
2. Group folders by SNC parent (e.g., all AGR together, all POL together)
3. For each SNC group with ≥3 folders in top-K:
   a. Present only that group's folders to the LLM
   b. Ask: "Given this query, rank ONLY these folders within the same classification"
   c. Use document summaries as evidence when available
4. Merge re-ranked groups back into final ranking (preserve inter-group ordering)
```

This is more token-efficient than full listwise re-ranking because each LLM call handles a small, focused cluster (e.g., 5-15 folders from the same SNC) rather than all 50-100 candidates at once.

#### PROMPT P3D-INTRAGROUP

```
You are an expert on U.S. State Department records on Brazil (1960s-1970s). A researcher 
is searching for specific content, and the retrieval system has identified a cluster of 
folders with similar classifications. Your task is to DISAMBIGUATE within this cluster.

RESEARCH TOPIC:
Title: {title}
Description: {description}

FOLDER CLUSTER (all classified under "{snc_parent_description}"):
{folder_cluster}

[Format per folder:]
[#{i}] ID: {folder_id}
  Label: {folder_label} | Period: {date_range}
  Documents Available: {yes — N docs / no}
  {IF docs: "Document Evidence:" + doc titles and summaries}
  Augmented Description: {augmented_description}

IMPORTANT: These folders all share a similar broad classification. Your job is to 
determine which specific folders are MOST LIKELY to contain content about the researcher's 
SPECIFIC topic — not just the broad category.

Consider:
1. Does the folder's date range match when the topic's events would have been active?
2. Does the folder label have geographic or topical qualifiers that match the query?
3. Do any available documents directly mention topics related to the query?
4. Based on your historical knowledge, which sub-classification is most relevant?

Rank ONLY the folders in this cluster from most to least relevant to the SPECIFIC query 
(not to the broad SNC category).

Return folder IDs in order, one per line:
1. {folder_id}
2. {folder_id}
...

REASON: [One sentence explaining what distinguishes the top-ranked folder from the others]
```

#### Experiment Set 3D

| Run ID | Base Phase 2 Run | LLM   | k  | Min Group Size | Mask | Compare Against             |
| ------ | ---------------- | ----- | -- | -------------- | ---- | --------------------------- |
| 3D-01  | Best 2A-R (Uni)  | {D15} | 50 | 3              | Uni  | 3C-03, 3B-03                |
| 3D-02  | Best 2A-R (Uni)  | {D15} | 50 | 2              | Uni  | 3D-01 (group size ablation) |
| 3D-03  | Best 2X (Uni)    | {D15} | 50 | best           | Uni  | 3D-01                       |

**Total 3D: 3 runs**

---

### Strategy 3E: Combined Pipeline (Precision + LLM) (NEW)

#### Why

The full pipeline uses each stage for what it does best:

```
Stage 1 (Phase 2):    Concept expansion → retrieve right SNC groups    [RECALL]
Stage 2 (3C):         Document evidence → score precision within groups [PRECISION]
Stage 3 (3D):         LLM disambiguation → resolve remaining ties      [REASONING]
```

#### Experiment Set 3E

| Run ID | Pipeline                        | Mask | Compare Against  |
| ------ | ------------------------------- | ---- | ---------------- |
| 3E-01  | Best 2A-R → 3C (best α) → 3D | Uni  | 3C-best, 3D-best |
| 3E-02  | Best 2X → 3C (best α) → 3D   | Uni  | 3E-01            |

**Total 3E: 2 runs**

**Phase 3 Total: 8 (3A) + 3 (3B) + 7 (3C) + 3 (3D) + 2 (3E) = 23 runs**

### Phase 3 Decisions (fill after running)

| Decision                                                         | Value                     |
| ---------------------------------------------------------------- | ------------------------- |
| D11 — Optimal k for re-ranking                                  | ___                       |
| D15 — LLM for re-ranking                                        | ___                       |
| D16 — Does 3B (LLM) outperform 3A (traditional) significantly?  | ☐ Yes / ☐ No            |
| D17 — Decay factor for no-doc folders in 3A                     | ___ (test 0.8, 0.9, 0.95) |
| D28 — Best alpha for 3C precision re-ranking                    | ___ (test 0.3, 0.5, 0.7)  |
| D29 — Does 3C outperform 3A significantly?                      | ☐ Yes / ☐ No            |
| D30 — Does 3D add value over 3C alone?                          | ☐ Yes / ☐ No            |
| D31 — Best min group size for 3D                                | ___ (test 2, 3)           |
| D32 — Does combined pipeline (3E) outperform individual stages? | ☐ Yes / ☐ No            |

---

## Master Run Registry

| Run ID             | Phase | Strategy                    | Query               | Model | Mask | nDCG@5 | ±CI  | vs Baseline* |
| ------------------ | ----- | --------------------------- | ------------------- | ----- | ---- | ------ | ----- | ------------ |
| **Baseline** | —    | TOFS+BCE+AllF+SimSNC        | Q0                  | BCE   | Uni  | 0.2141 | 0.049 | —           |
| 1A-01              | 1A    | AllF-F1                     | Q0                  | B     | —   |        |       |              |
| 1A-04              | 1A    | AllF-F1                     | Q0                  | BCE   | —   |        |       |              |
| 1A-11              | 1A    | AllF-F2                     | Q0                  | BCE   | —   |        |       |              |
| 1A-18              | 1A    | AllF-F3                     | Q0                  | BCE   | —   |        |       |              |
| 1A-25              | 1A    | AllF-F4                     | Q0                  | BCE   | —   |        |       |              |
| 1B-04              | 1B    | AllF                        | Q1                  | BCE   | —   |        |       |              |
| 1B-10              | 1B    | AllF                        | Q2                  | BCE   | —   |        |       |              |
| 1B-14              | 1B    | AllF                        | Q3                  | BCE   | —   |        |       |              |
| 1B-18              | 1B    | AllF                        | Q4                  | BCE   | —   |        |       |              |
| 1B-33              | 1B    | AllF                        | Q0+QALL             | BCE   | —   |        |       |              |
| **1BR-03**   | 1B-R  | AllF                        | **Q5**        | BCE   | —   |        |       |              |
| **1BR-06**   | 1B-R  | AllF                        | **Q6**        | BCE   | —   |        |       |              |
| **1BR-08**   | 1B-R  | AllF                        | **Q7**        | BCE   | —   |        |       |              |
| **1BR-10**   | 1B-R  | AllF                        | **Q2R**       | BCE   | —   |        |       |              |
| **1BR-17**   | 1B-R  | AllF                        | **Q0+QALL-R** | BCE   | —   |        |       |              |
| 2A-04              | 2A    | Aug-Base                    | Q0                  | BCE   | Uni  |        |       |              |
| 2A-11              | 2A    | Aug-Base                    | Q_best_BCE          | BCE   | Uni  |        |       |              |
| **2AR-01**   | 2A-R  | **SNC-Adaptive**      | Q0                  | BCE   | Uni  |        |       |              |
| **2AR-04**   | 2A-R  | **SNC-Adaptive**      | Q_best_R            | BCE   | Uni  |        |       |              |
| 2B-01              | 2B    | Aug-Homo                    | Q0                  | BCE   | Uni  |        |       |              |
| 2B-02              | 2B    | Aug-Homo                    | Q_best_BCE          | BCE   | Uni  |        |       |              |
| **2BR-01**   | 2B-R  | **SNC-Adapt+Homo**    | Q0                  | BCE   | Uni  |        |       |              |
| 2C-01              | 2C    | Aug-TCDE                    | Q0                  | BCE   | Uni  |        |       |              |
| 2C-03              | 2C    | Aug-TCDE                    | Q-TCDE              | BCE   | Uni  |        |       |              |
| **2T-03**    | 2T    | **Topic-Prior**       | Q0                  | BCE   | Uni  |        |       |              |
| **2T-06**    | 2T    | **Topic+SNC-Adapt**   | Q_best_R            | BCE   | Uni  |        |       |              |
| 2X-01              | 2X    | Hybrid                      | Q_best_BCE          | BCE   | Uni  |        |       |              |
| **2X-04**    | 2X    | **Best+Topic**        | Q_best_R            | BCE   | Uni  |        |       |              |
| 3A-04              | 3A    | Rerank-Trad                 | Q_best_BCE          | BCE   | Uni  |        |       |              |
| **3A-07**    | 3A    | Rerank-Trad                 | Q_best_R            | BCE   | Uni  |        |       |              |
| 3B-01              | 3B    | Rerank-LLM k=20             | Q_best_BCE          | BCE   | Uni  |        |       |              |
| **3B-03**    | 3B    | Rerank-LLM k=20             | Q_best_R            | BCE   | Uni  |        |       |              |
| **3C-03**    | 3C    | **Precision-Rerank**  | **Q0 (orig)** | BCE   | Uni  |        |       |              |
| **3D-01**    | 3D    | **LLM-IntraGroup**    | Q0                  | BCE   | Uni  |        |       |              |
| **3E-01**    | 3E    | **Combined-Pipeline** | Q0                  | BCE   | Uni  |        |       |              |

*Fill with significance marker (* = p<0.05 vs chosen comparison) after running.

---

## Implementation Timeline (Revised)

### Week 1-2: Infrastructure + Phase 0

- [ ] Implement `src/utils/llm_client.py` with disk-based caching (key by SHA256 of prompt)
- [ ] Implement `src/utils/results_manager.py` (save/load ranked lists keyed by run ID)
- [ ] Implement `src/utils/ecf_loader.py`
- [ ] Implement `src/utils/snc_classifier.py` (NEW — specific vs generic classification)
- [ ] Run Phase 0 data analysis scripts
- [ ] Fill decisions D6, D7, D8, D9

### Week 3: Phase 1 Query Generation (Original + Revised)

- [ ] Implement original 4 query generators; test on 3 topics before full run
- [ ] Implement revised 4 query generators (Q5, Q6, Q7, Q2R); test on 3 topics
- [ ] Run on all 45 topics → save all 8 query JSON files
- [ ] Implement `combine_queries.py` and `combine_queries_r.py`

### Week 3-4: Phase 1 Retrieval

- [ ] Implement `allf_index.py` (F1–F4 index builder)
- [ ] Implement `allf_retrieval.py` (supports B, E, C, BCE, BC, BE, CE)
- [ ] Run Set 1A (28 runs) → fill D1a, D1b, D1c, D1
- [ ] Run Set 1B (40 runs) → fill D2, D3, D4, D5
- [ ] Run Set 1B-R (19 runs) → fill D20, D21, D22, D23

### Week 5-6: Phase 2A + 2A-R

- [ ] Implement `aug_base.py` (original P2A prompts); test on 5 folders
- [ ] Implement `aug_snc_adaptive.py` (revised prompts with SNC classification); test on 5 specific + 5 generic folders
- [ ] Review and compare outputs: original vs revised
- [ ] Scale to all 5 Uniform ECFs (with caching)
- [ ] Run Set 2A (14 runs) and Set 2A-R (7 runs)

### Week 7: Phase 2B + 2B-R

- [ ] Implement `aug_homophily.py` (original); test on 5 empty folders
- [ ] Implement SNC-adaptive homophily (revised prompts); test on 5 specific + 5 generic empty folders
- [ ] Run Set 2B (8 runs) and Set 2B-R (4 runs)

### Week 8: Phase 2C + 2T

- [ ] Implement `aug_tcde.py` (two-pass, unchanged); test on 5 folders
- [ ] Implement topic modeling pipeline (BERTopic, no QRELs)
  - Build folder representations
  - Discover K topics (test K = 10, 15, 20)
  - Generate topic summaries
  - Implement query-topic classification
- [ ] Run Set 2C (9 runs) and Set 2T (6 runs)
- [ ] Run 2X cross-strategy (4 runs)

### Week 9-10: Phase 3

- [ ] Implement `rerank_traditional.py`; run Set 3A (8 runs)
- [ ] Implement `rerank_llm.py` with P3B-RERANK; run Set 3B (3 runs)
- [ ] Implement `rerank_precision.py` (3C — document-evidence precision with Q0 as re-rank query)
  - Test alpha values: 0.3, 0.5, 0.7
  - Implement borrowed neighbor evidence for folders without docs
  - Run Set 3C (7 runs)
- [ ] Implement `rerank_llm_precision.py` (3D — cluster-then-rank LLM disambiguation)
  - Group top-K by SNC parent
  - Test min group sizes: 2, 3
  - Run Set 3D (3 runs)
- [ ] Run combined pipeline 3E (2 runs)
- [ ] Fill D11, D15, D16, D17, D28, D29, D30, D31, D32

### Week 10: Analysis + Writing

- [ ] Run Wilcoxon signed-rank tests for all key pairwise comparisons
- [ ] Build comparison table: Phase 1 best → Phase 2 best → Phase 3 best → Baseline
- [ ] **Key comparison:** Original vs Revised (entity-focused vs concept-focused) at every phase
- [ ] Analyze per-topic: identify which formerly hard topics are now resolved
- [ ] Analyze by folder class: with-docs vs without-docs nDCG@5
- [ ] **Analyze by SNC specificity:** specific vs generic SNC performance (NEW)
- [ ] **Analyze topic-prior impact:** which topics benefit most from topic modeling (NEW)
- [ ] Write dissertation sections

---

## Future Decisions Tracker

| #             | Decision                                            | Needed Before        | Options                                                    | Fill-In                                                 |
| ------------- | --------------------------------------------------- | -------------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| D1            | F_best (best folder index config, overall)          | Phase 1B             | F1, F2, F3, F4,**F5**                                | ___                                                     |
| D1a           | F_best for B                                        | Phase 1B             | —                                                         | ___                                                     |
| D1b           | F_best for E                                        | Phase 1B             | —                                                         | ___                                                     |
| D1c           | F_best for C                                        | Phase 1B             | —                                                         | ___                                                     |
| D2            | Q_best_B (original)                                 | Phase 2A             | Q0, Q1, Q2, Q4, Q12, Q14                                   | ___                                                     |
| D3            | Q_best_E (original)                                 | Phase 2A             | Q0, Q1, Q3, Q13                                            | ___                                                     |
| D4            | Q_best_C (original)                                 | Phase 2A             | Q0, Q1, Q4, Q14                                            | ___                                                     |
| D5            | Q_best_BCE (original)                               | Phase 2A             | Any combination                                            | ___                                                     |
| D6            | Homophily similarity criterion                      | Phase 2B             | **Cascading hierarchy** (exact → 2-level → parent) | **Cascading hierarchy** — *Phase 0 pre-filled* |
| D7            | Homophily min neighbor docs threshold               | Phase 2B             | 1 / 2 / 3 / 5                                              | ___ (suggested: 3)                                      |
| D8            | Max neighbor docs in P2B-HOMO prompt                | Phase 2B             | 3 / 5 / 10                                                 | ___ (suggested: 5)                                      |
| D9            | Proceed with Homophily strategy?                    | Phase 2B             | **Yes** — 73.8% viability at parent level           | **Yes** — *Phase 0 pre-filled*                 |
| D10           | Aug_best (best augmentation, Uniform)               | Phase 3              | 2A / 2A-R / 2B / 2B-R / 2C / 2T / Hybrid                   | ___                                                     |
| D11           | Optimal k for re-ranking                            | Phase 3              | 20 / 50 / 100                                              | ___                                                     |
| D12           | N topics per folder for TCDE                        | Week 8 pilot         | 3 / 5 / 7                                                  | ___                                                     |
| D13           | Document content for augmentation                   | Week 5 pilot         | title+summary / title+OCR-p1 / full-OCR                    | ___                                                     |
| D14           | LLM model for augmentation                          | Week 1               | GPT-4o / Claude Sonnet / Claude Opus                       | ___                                                     |
| D15           | LLM model for re-ranking                            | Week 9               | GPT-4o / Claude Sonnet                                     | ___                                                     |
| D16           | Does 3B (LLM rerank) outperform 3A?                 | After Phase 3        | Yes / No                                                   | ___                                                     |
| D17           | Decay factor for no-doc folders in 3A               | Week 9               | 0.80 / 0.90 / 0.95                                         | ___                                                     |
| **D18** | **Unknown-SNC fallback strategy**             | **Phase 1**    | **label-as-SNC / main_title / box-neighbors / skip** | ___ (recommended: label + main_title)                   |
| **D19** | **Max docs per folder in homophily sampling** | **Phase 2B**   | **1 / 2 / 3**                                        | ___ (recommended: 2)                                    |
| **D20** | **Q_best_B (revised or original?)**           | **Phase 2A**   | **Q0, Q5, Q6, Q2R, Q0+Q5, Q0+Q6, Q0+QALL-R**         | ___                                                     |
| **D21** | **Q_best_E (revised or original?)**           | **Phase 2A**   | **Q0, Q5, Q7, Q0+Q5, Q0+QALL-R**                     | ___                                                     |
| **D22** | **Q_best_BCE (revised or original?)**         | **Phase 2A**   | **Any revised or original combination**              | ___                                                     |
| **D23** | **Revised > Original? (overall)**             | **After 1B-R** | **Yes / No / Mixed**                                 | ___                                                     |
| **D24** | **SNC-Adaptive > Original prompts?**          | **After 2A-R** | **Yes / No**                                         | ___                                                     |
| **D25** | **Topic Prior improves ranking?**             | **After 2T**   | **Yes / No**                                         | ___                                                     |
| **D26** | **Best alpha for Topic Prior**                | **After 2T**   | **0.1 / 0.2 / 0.3**                                  | ___                                                     |
| **D27** | **Best K (number of topics)**                 | **After 2T**   | **10 / 15 / 20**                                     | ___                                                     |

---

## LLM Cost and Caching Notes

### Caching Strategy

- All LLM calls cached on disk keyed by `SHA256(prompt_text)`
- Folders without docs → augmented once, reused across all ECFs (saves ~50% of 2A/2C calls)
- Folders with docs → cached by `SHA256(folder_id + sorted(doc_ids))` — same doc set = same cache hit
- Phase 1 query generation: 45 topics × 8 types = **360 calls** (very cheap; run once)

### Estimated Call Volumes

| Component                         | Estimated LLM Calls                                                      | Notes                                          |
| --------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------- |
| Phase 1 query gen (orig+rev)      | 360                                                                      | Fixed, cheap                                   |
| Phase 2A augmentation (original)  | ~1,336 × 2 (doc + nodoc) × ECF variation (with caching ~40% reduction) | Largest cost                                   |
| Phase 2A-R augmentation (revised) | ~1,336 × 4 prompts × ECF variation (with caching)                      | 4 prompts instead of 2, but similar cache rate |
| Phase 2B augmentation             | ~668 empty folders × Homo prompt (with caching)                         | Less than 2A                                   |
| Phase 2B-R augmentation           | ~668 × 2 prompts (specific/generic)                                     | Slightly more than 2B                          |
| Phase 2C augmentation             | ~1,336 × 2 passes × ECF variation (with caching)                       | 2× Phase 2A                                   |
| Phase 2T topic modeling           | ~45 classify calls + ~K summary calls                                    | Very cheap                                     |
| Phase 3B re-ranking               | 45 topics × 5 ECFs × 1 call per k folders                              | Depends on k                                   |

**Recommendation:** Run Phase 2A and 2A-R pilots on 1 ECF first to compare original vs revised prompt quality. Scale to all 5 ECFs only after confirming the revised prompts produce better output.

### Ablation Studies (within Phase 2A-R)

- DESCRIPTION only vs KEYWORDS only vs both in augmented label
- Original folder label preserved in index vs replaced entirely
- Top-3 docs vs top-5 docs vs all available docs per folder
- **Specific prompt for all vs Generic prompt for all vs Split (2AR-06, 2AR-07 ablations)**
- These can be run as sub-experiments within Phase 2A-R before scaling to all 5 ECFs

---

## Summary of Revisions from Original experiments.md

| Component                   | Original                             | Revised                                           | Rationale                                                    |
| --------------------------- | ------------------------------------ | ------------------------------------------------- | ------------------------------------------------------------ |
| **Q1 (HyDE)**         | Generate fake SNC codes              | Q6: Folder descriptions in real SNC label style   | Fake codes don't exist in the index                          |
| **Q2 (Keywords)**     | Entity/figure names                  | Q2R + Q5: Concept terms and category bridging     | Entities don't appear in folder labels; queries match themes |
| **Q3 (EmbText)**      | Event-focused paragraph              | Q7: Concept/term-focused thematic paragraph       | Queries are generic themes, not events                       |
| **P2A (Folder Aug)**  | Same prompt for all SNCs             | Split: specific-SNC vs generic-SNC prompts        | POL needs documents; AGR needs concepts                      |
| **P2A Keywords**      | "key_figures", "historical_entities" | Domain concepts, related terms, sub-topics        | Queries match themes, not people                             |
| **P2B (Homophily)**   | Uniform neighbor treatment           | Document weight depends on SNC specificity        | For generic SNCs, docs are primary signal                    |
| **Topic Modeling**    | Not present                          | Phase 2T: Unsupervised topic discovery (no QRELs) | Collection structure helps ranking                           |
| **Phase 3**           | 6+2 runs                             | 8+3 runs (added revised strategy comparisons)     | Need to evaluate revised pipeline end-to-end                 |
| **Total experiments** | ~75+33 (108)                         | ~87+52+6+4+11+3 =**163 total runs**         | More comprehensive with ablations                            |
