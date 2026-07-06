# Phase 0 Data Findings → Experiment Design Changes

**Author:** Victor Hugo Oliveira Leão
**Date:** 2026-06-28
**Based on:** Dashboard analysis of `FoldersV1.3.json` (1,336 folders) and `itemsV1.2.json` (31,681 documents)

---

## Summary of Key Findings

| Finding                        | Quantitative Evidence                                                   | Impact                                        |
| ------------------------------ | ----------------------------------------------------------------------- | --------------------------------------------- |
| Unknown SNC folders            | 68 folders (5.1%), 1,328 docs,**empty** `label_parent_expanded` | Breaks F2/F4 configs & augmentation prompts   |
| POL dominance                  | 353 folders (26.4%), 11,432 docs under parent`POL`                    | Wasteful to use all docs for augmentation     |
| Hierarchical homophily gains   | Exact: 69.9% → 2-Level: 76.9% → Parent:**89.9%** (≥1 doc)      | Must cascade up SNC hierarchy                 |
| Scope notes sparse             | Only 57/379 SNCs (15%), covering 504/1,336 folders (37.7%)              | F3/F4 limited; prompts need conditional logic |
| Missing end dates              | 1,008/1,336 folders (75.4%) have "Unknown" endDate                      | Prompts must handle gracefully                |
| Empty`label_parent_expanded` | 190 folders (14.2%), all from Unknown/top-level SNCs                    | F2/F4 produce empty fields for these          |

---

## Change 1: Handle Unknown-SNC Folders (68 folders, 1,328 docs)

### Evidence

- 68 folders have `snc = "Unknown"`, `parent = ""`, `label_parent_expanded = ""`
- These folders contain **1,328 documents** (4.2% of the collection) — non-trivial
- Their `label` field contains meaningful descriptors:
  - `"USAID-NE SECRET 1964-67"`, `"Economic Affairs(General)"`, `"Defense"`, `"Buildings & Grounds"`, `"Education 1966"`, `"Intelligence"`, etc.
- Without handling, these folders are **invisible** to F2/F4 configurations and receive empty context in augmentation prompts

### Changes to `experiments.md`

**Section: Folder Index Configurations (Phase 1)**

Add a new folder index configuration **F5** and update the F2/F4 definitions.

```diff
 ### Folder Index Configurations (Phase 1)
 | Code | Content indexed per folder |
 |------|---------------------------|
 | F1 | Folder Label only |
-| F2 | Folder Label + Parent Expanded SNC |
+| F2 | Folder Label + Parent Expanded SNC (uses label as fallback when parent_expanded is empty) |
 | F3 | Folder Label + Scope Note (blank for folders without scope note) |
-| F4 | Folder Label + Parent Expanded SNC + Scope Note |
+| F4 | Folder Label + Parent Expanded SNC + Scope Note (uses label as fallback when parent_expanded is empty) |
+| F5 | Folder Label + Parent Expanded SNC + Scope Note + main_title (always populated) |
```

**Section: All augmentation prompts (P2A-DOC, P2A-NODOC, P2B-HOMO, P2C-TOPIC)**

```diff
 FOLDER INFORMATION:
 - Folder Label: {folder_label}
-- SNC Code: {snc}
-- SNC Meaning: {expanded_snc}
-- Broader Category: {parent_expanded_snc}
+- SNC Code: {snc} (or "Unclassified" if Unknown)
+- SNC Meaning: {expanded_snc} (or derived from folder label if SNC is Unknown)
+- Broader Category: {parent_expanded_snc} (or inferred from folder label if empty)
```

**Add implementation note after Phase 1 configs:**

```
> **Data-Driven Note — Unknown SNC Handling:**
> 68 folders (5.1%) have SNC="Unknown" with empty label_parent_expanded.
> Their `label` field contains meaningful text (e.g., "Economic Affairs(General)",
> "Intelligence", "Education 1966"). For these folders:
> 1. F2/F4 configs: use `label` + `main_title` as the expanded SNC substitute
> 2. Augmentation prompts: derive SNC meaning from the folder label text
> 3. Homophily: match by `main_title` similarity or `box` proximity instead of SNC
> This ensures these 1,328 documents remain findable.
```

### New Decision

| #   | Decision                      | Needed Before | Options                                          | Fill-In |
| --- | ----------------------------- | ------------- | ------------------------------------------------ | ------- |
| D18 | Unknown-SNC fallback strategy | Phase 1       | label-as-SNC / main_title / box-neighbors / skip | ___     |

---

## Change 2: Document Sampling for Large SNCs (POL dominance)

### Evidence

- `POL` parent: 353 folders, 11,432 docs (36.1% of all documents)
- `POL 15` alone: 64 folders, 2,582 docs
- `POL 23`: 37 folders, 1,882 docs
- Some individual SNCs like `POL 2` have 52 folders
- Using all documents from large SNCs in augmentation prompts is: (a) wasteful of LLM tokens, (b) redundant, (c) expensive

### Changes to `experiments.md`

**Section: Strategy 2A — Document Content to Use (line ~421)**

```diff
 #### Document Content to Use
-Use `title + summary` per document (GPT-4o summaries already exist in the collection).
-If a folder has >5 training documents, use the top-5 by BM25 score against the folder
-label (as a quick proxy for most representative). Fill decision D13 after reviewing pilot outputs.
+Use `title + summary` per document (GPT-4o summaries already exist in the collection).
+
+**Document selection strategy (data-driven):**
+- If a folder has ≤5 training documents: use all of them
+- If a folder has >5 training documents: select **2 random documents from each of the
+  folder's training docs** plus the top-3 by BM25 score against the folder label (capped at 5 total)
+- **For SNC-level homophily (Phase 2B):** when an SNC has many folders with docs,
+  select at most **2 random documents from each folder** within that SNC (capped at D8 total).
+  This ensures diversity across folders rather than depth within one folder.
+
+Fill decision D13 after reviewing pilot outputs.
```

**Section: Strategy 2B — Code to Implement (line ~563)**

```diff
 src/augmentation/aug_homophily.py
-  ├── find_neighbor_folders(folder_id, snc, box_id, all_folders_with_docs, strategy=D6)
+  ├── find_neighbor_folders(folder_id, snc, box_id, all_folders_with_docs, strategy=D6, max_docs_per_folder=2)
   │     → [(neighbor_folder_id, doc_ids), ...]
-  ├── select_neighbor_docs(neighbor_list, max_docs=D8)
+  ├── select_neighbor_docs(neighbor_list, max_docs=D8, max_per_folder=2)
   │     → [doc_id, ...]
+  │     # For large SNCs (>10 folders with docs): sample 2 docs from each of up to
+  │     # D8/2 distinct folders. Ensures SNC-level diversity over single-folder depth.
```

### New Decision

| #   | Decision                                  | Needed Before | Options   | Fill-In      |
| --- | ----------------------------------------- | ------------- | --------- | ------------ |
| D19 | Max docs per folder in homophily sampling | Phase 2B      | 1 / 2 / 3 | Suggested: 2 |

---

## Change 3: Hierarchical Homophily with Cascading Fallback

### Evidence

The data shows a **dramatic improvement** when cascading up the SNC hierarchy:

| Match Level           | ≥1 neighbor doc | ≥3 neighbor docs |
| --------------------- | ---------------- | ----------------- |
| **Exact SNC**   | 541 (69.9%)      | 333 (43.0%)       |
| **2-Level SNC** | 595 (76.9%)      | 426 (55.0%)       |
| **Parent SNC**  | 696 (89.9%)      | 571 (73.8%)       |

Going from exact SNC to parent level recovers an additional **155 empty folders** (20% more) with ≥1 neighbor doc, and **238 more** with ≥3. This means homophily is clearly viable at the parent level.

### Changes to `experiments.md`

**Section: Strategy 2B — Decisions to Fill (line ~552)**

```diff
 #### Decisions to Fill (from Phase 0)
-- D6 — Similarity criterion: ☐ Same SNC (exact) / ☐ Same SNC 2-level prefix / ☐ Same Box / ☐ Both: ___
+- D6 — Similarity criterion: **☑ Cascading hierarchy** (recommended based on Phase 0 data):
+  1. First try exact SNC match → if ≥ D7 docs found, use them
+  2. If insufficient, expand to 2-level SNC prefix → if ≥ D7 docs found, use them
+  3. If still insufficient, expand to parent SNC → if ≥ D7 docs found, use them
+  4. If still insufficient, fall back to P2A-NODOC (pure knowledge prompt)
+  Record which level was used for each folder (for analysis).
 - D7 — Minimum neighbor docs threshold: ___ (suggested: 3)
 - D8 — Max neighbor docs in prompt: ___ (suggested: 5)
-- D9 — Proceed with Homophily? ☐ Yes / ☐ No (if <20% of empty folders pass threshold) / ☐ Hybrid: ___
+- D9 — Proceed with Homophily? **☑ Yes** — with cascading hierarchy, 73.8% of empty folders
+  have ≥3 parent-level neighbor docs, well above the 20% viability threshold.
```

**Section: Strategy 2B — Code to Implement (line ~560)**

```diff
 src/augmentation/aug_homophily.py
-  ├── find_neighbor_folders(folder_id, snc, box_id, all_folders_with_docs, strategy=D6)
+  ├── find_neighbor_folders(folder_id, snc, snc_2level, parent, box_id, all_folders_with_docs)
   │     → [(neighbor_folder_id, doc_ids), ...]
+  │     # Cascading: exact_snc → snc_2level → parent → box → empty
+  │     # Returns (neighbors, match_level) for logging
```

**Section: P2B-HOMO prompt (line ~579)** — update the context description:

```diff
 CONTEXTUAL DOCUMENTS FROM RELATED FOLDERS:
-The documents below come from OTHER folders with the same SNC classification ({snc}).
+The documents below come from OTHER folders with a related SNC classification
+(match level: {match_level} — e.g., exact "{snc}", 2-level "{snc_2level}", or parent "{parent}").
 They do NOT belong to the target folder. Use them to understand what topics, actors,
 and events are typically filed under this code — but do not assume their specific
 events apply to the target folder.
```

### Pre-filled Decisions

```
D6 = Cascading hierarchy (exact → 2-level → parent)
D9 = Yes (73.8% viability at parent level with threshold 3)
```

---

## Change 4: Scope Note Handling — Conditional Logic in Prompts

### Evidence

- Only **57 of 379 distinct SNCs** (15%) have scope notes
- These cover **504 of 1,336 folders** (37.7%)
- F3 and F4 configurations will have **empty scope note fields for 62.3% of folders**
- The `experiments.md` already notes this (line 304: "F3/F4 effects are limited to 50/379 SNCs") but the prompt templates don't handle it

### Changes to `experiments.md`

**Section: Augmented Label Index Format (line ~407)**

```diff
 ### Augmented Label Index Format
 After generation, each folder's entry in the retrieval index will be:
-{original_folder_label} | {expanded_snc} | {DESCRIPTION} | {KEYWORDS}
+{original_folder_label} | {expanded_snc} | {scope_note_if_available} | {DESCRIPTION} | {KEYWORDS}
-The original label is preserved as a prefix (exact-match fallback) while the enriched content follows.
+The original label is preserved as a prefix (exact-match fallback) while the enriched content follows.
+Scope note is included when available (37.7% of folders). For the remainder, this field is omitted.
```

**Section: All prompts — add conditional scope note**

```diff
 FOLDER INFORMATION:
 - Folder Label: {folder_label}
 - SNC Code: {snc}
 - SNC Meaning: {expanded_snc}
 - Broader Category: {parent_expanded_snc}
-- Scope Note: {scope_note}
+- Scope Note: {scope_note if available, otherwise "No scope note available for this SNC code."}
 - Date Range: {start_date} to {end_date}
 - Record Group: {record_group}
```

**Add note after Phase 1 Set 1A (line ~304):**

```diff
-☐ **After 1A:** Record F_best for B, E, C, BCE individually. Note: F3/F4 effects are limited to 50/379 SNCs that have scope notes — may not show strong gains.
+☐ **After 1A:** Record F_best for B, E, C, BCE individually.
+> **Data-Driven Note — Scope Notes:** Only 57/379 SNCs (15%) have scope notes, covering
+> 504/1,336 folders (37.7%). F3/F4 may show modest gains overall but could significantly
+> help the 37.7% of folders with scope notes. Consider computing nDCG@5 separately for
+> folders with vs. without scope notes to measure targeted impact.
```

---

## Change 5: Missing End Dates — Prompt Robustness

### Evidence

- **1,008 of 1,336 folders (75.4%)** have `endDate = "Unknown"`
- All 1,336 folders have valid start dates (no missing start dates)
- The augmentation prompts use `{start_date} to {end_date}` which would produce `"01/01/1964 to Unknown"` for 75% of folders
- This affects the LLM's ability to reason about the time period

### Changes to `experiments.md`

**Section: All prompts that use date ranges**

```diff
-- Date Range: {start_date} to {end_date}
+- Date Range: {start_date}{" to " + end_date if end_date != "Unknown" else " (end date not recorded)"}
```

**In P2A-NODOC HISTORICAL CONTEXT (line ~492):**

```diff
-This folder is from U.S. State Department records on Brazil. The period {start_date}–{end_date} includes:
+This folder is from U.S. State Department records on Brazil, starting from {start_date}.
+{If end_date is known: "The period covers " + start_date + "–" + end_date + "."}
+{If end_date is Unknown: "The end date is not recorded, but based on the SNC classification
+and folder context, content likely extends through the late 1960s or early 1970s."}
+The broader historical period includes:
 the April 1964 military coup and subsequent military government, Cold War dynamics in
 Latin America, significant Brazilian economic development, and evolving U.S.-Brazil
 bilateral relations across security, trade, and diplomacy.
```

**Add to Phase 0 Required Outputs:**

```diff
 ### Required Outputs from Phase 0
 1. SNC coverage table
 2. Homophily viability number
 3. Filled decisions D6, D7, D8, D9
 4. Insights about the SNCs and how they work.
 5. Insights about the documents and their relation to the SNC.
+6. **Date completeness report:** 75.4% of folders lack end dates — prompts must handle gracefully.
+7. **Unknown SNC report:** 68 folders (5.1%) with 1,328 docs have no SNC classification.
```

---

## Change 6: `label_parent_expanded` Quality Issues

### Evidence

- **190 folders (14.2%)** have empty or missing `label_parent_expanded`
- All 68 Unknown-SNC folders have empty `label_parent_expanded`
- Some top-level SNCs (e.g., `AID`, `DEF`, `FN`) have `label_parent_expanded` that just repeats the SNC name with no enrichment (e.g., `SNC="AID"` → `label_parent_expanded="AID"`)
- Multi-level SNCs have rich expansions: `SNC="POL 18"` → `"POLITICAL AFFAIRS & RELATIONS: PROVINCIAL, MUNICIPAL & STATE GOVERNMENT"`

### Changes to `experiments.md`

**Section: Folder Index Configurations — add notes:**

```markdown
> **Data-Driven Note — label_parent_expanded Quality:**
> - 190 folders (14.2%) have empty `label_parent_expanded` (all Unknown-SNC + some top-level SNCs)
> - Top-level SNCs (AID, DEF, FN, etc.) have minimal expansion — e.g., "AID" → "AID"
> - Multi-level SNCs have rich expansions — e.g., "POL 18" → "POLITICAL AFFAIRS & RELATIONS: PROVINCIAL, MUNICIPAL & STATE GOVERNMENT"
> - **Implication for F2/F4:** These configurations provide the most benefit for multi-level SNCs
>   where the expansion adds real information. For top-level and Unknown SNCs, F2 ≈ F1.
> - **Implementation:** When `label_parent_expanded` is empty or equals the SNC code,
>   fall back to `main_title` field (always populated) as the expansion.
```

---

## Summary of New/Pre-filled Decisions

| #   | Decision                                  | Pre-filled Value                                           | Rationale                                        |
| --- | ----------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------ |
| D6  | Homophily similarity criterion            | **Cascading hierarchy** (exact → 2-level → parent) | Parent level reaches 89.9% of empty folders      |
| D9  | Proceed with Homophily?                   | **Yes**                                              | 73.8% viability at parent level with threshold 3 |
| D18 | Unknown-SNC fallback strategy             | ___ (recommend: label + main_title)                        | 68 folders would be invisible without this       |
| D19 | Max docs per folder in homophily sampling | ___ (recommend: 2)                                         | Prevents POL-dominant SNCs from flooding prompts |

## Sections of `experiments.md` Affected

| Section                         | Lines                     | Type of Change                                       |
| ------------------------------- | ------------------------- | ---------------------------------------------------- |
| Folder Index Configs (Phase 1)  | 35-41                     | Add F5, update F2/F4 descriptions                    |
| Phase 0 Checklist               | 63-111                    | Add required outputs#6, #7                           |
| After Phase 1 Set 1A            | 304                       | Expand scope note warning                            |
| Strategy 2A — Document Content | 421-422                   | Add sampling strategy                                |
| Augmented Label Index Format    | 407-412                   | Add scope note field                                 |
| All augmentation prompts        | 446-507, 579-608, 676-721 | Handle Unknown SNC, missing dates, conditional scope |
| Strategy 2B — Decisions        | 552-556                   | Pre-fill D6, D9, add cascading logic                 |
| Strategy 2B — Code             | 560-575                   | Add hierarchical search, per-folder sampling         |
| Future Decisions Tracker        | 1004-1028                 | Add D18, D19, pre-fill D6, D9                        |
| LLM Cost Notes                  | 1032-1055                 | Update estimates with sampling reduction             |
