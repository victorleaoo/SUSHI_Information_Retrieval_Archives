# All-Folders (ALLFL) Augmented Experiment Runs

Tracks the experiment set driven by `run_all_folders_augmented_experiments.py`: the
"all folders" (ALLFL) `CONFIGS` entries from `run_new_experiments.py` -- every config
whose suffix ends in `.b` or `.c`, i.e. every config that touches the ALLFL ("all
folders label") ranker, either as the hybrid RRF partner or as the ranker itself --
re-run for every query field (T / TD / TDN), in two query variants, with the ALLFL
ranker's folder content always sourced from the augmented folder label.

## Prerequisites

This script **consumes** the output of two LLM experiments; it does not generate
either of them:

- `src.llm_experiments.query_expansion.core_themes_and_related_concepts.generate`
  → `data/llm_calls/query_expansion/2_core_themes_and_related_concepts/{T,TD,TDN}.json`
  (per-topic `core_themes` + `related_concepts`, used for the `CT` query variant)
- `src.llm_experiments.folder_label_augmentation.generate`
  → `data/llm_calls/folder_label_augmentation/2_folder_context/folders.json`
  (per-folder `core_themes` + `related_concepts`, keyed by folder ID, used for the
  augmented ALLFL folder content in *every* run of this script)

## The "all folders" configs

7 `CONFIGS` offsets (out of 17) touch the ALLFL ranker:

| Offset | Suffix | Role |
|---|---|---|
| 1 | `V.TOFS.mx--2.b` | Tuned BM25F + ColBERT (TOFS), hybrid RRF with the ALLFL BM25 partner |
| 2 | `X.TOFS.mx--2.b` | Untuned BM25F + ColBERT (TOFS), hybrid RRF with the ALLFL BM25 partner |
| 9 | `-.----.-.c` | ALLFL ColBERT baseline (ranker itself, no main-ranker fields) |
| 14 | `W.TOFS.-.c` | W (TOFS), hybrid RRF with the ALLFL ColBERT partner |
| 15 | `W.TOFS.s---2.c` | W (TOFS) + similar_snc k=2, hybrid RRF with the ALLFL ColBERT partner |
| 16 | `-.----.-.b` | ALLFL untuned-BM25 baseline (ranker itself) -- **new**, added alongside this script |
| 17 | `-.----.-.e` | ALLFL Embeddings baseline (ranker itself) -- **new**, added alongside this script |

Offsets 16 and 17 are also available un-augmented via `run_new_experiments.py`
directly (ids 16/46/76 and 17/47/77) -- they mirror offset 9's ALLFL-baseline
shape for the two model types that didn't have one yet.

## Query variants

| Variant | Composition |
|---|---|
| `STD` | The plain query (title / title+description / title+description+narrative), no augmentation |
| `CT`  | `STD` query + `core_themes` paragraph + `", ".join(related_concepts)`, from the topic's `core_themes_and_related_concepts` entry |

Built by `run_generator.build_core_themes_augmentation_text(topic_entry)` and applied
via `RunGenerator.get_augmented_query()`, the same pattern as the `DOC`/`FL`/`AUG`
variants in `run_query_augmentated_experiments.py`
(`f"{original_query}. {augmentation_text}"`).

## Folder-label augmentation (always on)

In **both** query variants, the ALLFL ranker's folder-level content is not just the
plain `folderMetadata` label -- it's the label plus the folder's augmented
`core_themes` + `related_concepts` text:

```python
label = folderMetadata[folder]['label_parent_expanded'] + " " + folderMetadata[folder]['scope_truncated']
aug = folders_json[folder]['parsed']['core_themes'] + " " + ", ".join(folders_json[folder]['parsed']['related_concepts'])
folder_content = f"{label}. {aug}"
```

This is controlled by `RunGenerator`'s `all_folders_folder_label_augmented=True` flag
(set unconditionally by this script), built by
`run_generator.build_folder_label_augmentation_text(folder_entry)`. It's a fixed
property of every run in this script -- not a 3rd axis -- which is exactly what
distinguishes these runs from the plain ALLFL runs already in `all_runs/` produced by
`run_new_experiments.py`.

## Known limitation: ColBERT query truncation

Same caveat as `runs.md`: ColBERT's query encoder truncates queries to ~32 tokens.
5 of the 7 configs involve ColBERT (1, 2, 9, 14, 15); the `CT` variant's augmentation
text is likely to be mostly or entirely truncated for those. Per the existing project
decision, run them anyway rather than excluding them -- a lack of effect in a
ColBERT-containing config's `CT` row may reflect truncation, not a finding about the
augmentation itself. The two new baselines (16 BM25, 17 Embeddings) aren't affected.

## Naming convention

The `STD` variant keeps the plain `run_new_experiments.py` query-field tag
(`T--`/`TD-`/`TDN`) unchanged. The `CT` variant uses the same 4th-slot trick as
`run_query_augmentated_experiments.py` (drop the base tag's last char, append `CT`):
`T--CT` / `TD-CT` / `TDNCT`. Every folder name additionally gets a trailing `.FLAug`
suffix marking "ALLFL content is the augmented folder label":

```
U5.T--.V.TOFS.mx--2.b.FLAug      (T,   STD, offset 1)
U5.T--CT.V.TOFS.mx--2.b.FLAug    (T,   CT,  offset 1)
U5.TDNCT.-.----.-.e.FLAug        (TDN, CT,  offset 17)
```

## Scope

- 7 `CONFIGS` offsets (1, 2, 9, 14, 15, 16, 17).
- All 3 query fields: T, TD, TDN.
- 2 query variants: STD, CT.
- 7 × 3 × 2 = **42 experiments**.
- `U5` only (default `docs_per_box=5`, uniform sampling) -- no `DOCSBOX` sweep, no
  `OFFICIAL` ECF re-runs.
- Standard `random` protocol, 30 seeds (`RANDOM_SEED_LIST`).
- Hybrid configs (offsets 1, 2, 14, 15) apply the *same* query variant to both the
  main ranker and the ALLFL partner ranker used in the RRF fusion, and the augmented
  folder label always applies to the ALLFL partner regardless of variant.

## Running

From the project root:

```bash
python -m src.llm_experiments.runs.run_all_folders_augmented_experiments              # all 42
python -m src.llm_experiments.runs.run_all_folders_augmented_experiments 2001 2002    # specific ids
python -m src.llm_experiments.runs.run_all_folders_augmented_experiments T TD         # every config/variant for those query fields
python -m src.llm_experiments.runs.run_all_folders_augmented_experiments CT           # every config/query-field for that variant only
```

Output lands in `all_runs/<name>/` (same layout/marker files as
`run_new_experiments.py`); an experiment already completed there is skipped
automatically (`is_experiment_done`).

## Experiment ID scheme

`id = 2000 + query_field_offset + variant_offset + config_offset`

- `query_field_offset`: T=0, TD=100, TDN=200
- `variant_offset`: STD=0, CT=30
- `config_offset`: one of 1, 2, 9, 14, 15, 16, 17 (matches `CONFIGS` offsets in
  `run_new_experiments.py`; gaps are expected)

Range: 2001–2247.

## Run-tracking table

Fill in dates/status as batches complete. One row per (query field, variant); each
row covers 7 configs (30 seeds each, except offsets 9/16/17 which are seed-independent
single-pass ALLFL runs) = up to 210 seed-runs.

| Query field | Variant | ID range        | Status | Notes |
|---|---|---|---|---|
| T   | STD | 2001–2017 (sparse) | not started | |
| T   | CT  | 2031–2047 (sparse) | not started | |
| TD  | STD | 2101–2117 (sparse) | not started | |
| TD  | CT  | 2131–2147 (sparse) | not started | |
| TDN | STD | 2201–2217 (sparse) | not started | |
| TDN | CT  | 2231–2247 (sparse) | not started | |
