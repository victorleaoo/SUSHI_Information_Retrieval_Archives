# Query-Augmentation Experiment Runs

Tracks the experiment set driven by `run_query_augmentated_experiments.py`: the
15 core configurations from `run_new_experiments.py`, re-run for every query
field (T / TD / TDN), against three LLM-based query-augmentation variants
built from the `query_expansion/doc_folder_hip` experiment's output.

## Prerequisite: doc_folder_hip data

This script **consumes** the output of
`src.llm_experiments.query_expansion.doc_folder_hip.generate` — it does not
generate it. That data must already exist at:

```
data/llm_calls/query_expansion/1_doc_folder_hip/{T,TD,TDN}.json
```

Each file is a dict keyed by topic ID (`T18Eval-XXXXX`, matching the ECF
`Topics` keys used by `RunGenerator`). Each topic entry looks like:

```json
{
  "topic_id": "T18Eval-00002",
  "original_query": {"TITLE": "...", "DESCRIPTION": "..."},
  "query_text": "...",
  "documents": {
    "raw_response": "...",
    "parsed": {"doc_1": "...", "doc_2": "...", "doc_3": "..."},
    "input_tokens": 245, "output_tokens": 165, "duration_seconds": 1.84,
    "created_at": "..."
  },
  "folder_label": {
    "raw_response": "...",
    "parsed": {
      "snc_codes": ["DEF 19-8", "POL 15-1"],
      "label_text": ["DEF 19-8 Naval Forces & Equipment BRAZ 1969-1973", "..."],
      "subject_terms": ["Brazilian Navy", "submarines", "..."]
    },
    "input_tokens": 260, "output_tokens": 130, "duration_seconds": 1.51,
    "created_at": "..."
  }
}
```

(Full details in `src/llm_experiments/README.md`, under `query_expansion/doc_folder_hip/`.)

The T/TD/TDN files were generated on a separate machine and are expected to
already be present under `data/llm_calls/query_expansion/1_doc_folder_hip/`
before running this script; it does not launch any LLM calls itself.

## Augmentation variants

Built by `run_generator.build_augmentation_text(topic_entry, variant)`:

| Variant | Composition |
|---|---|
| `DOC` | `doc_1`. `doc_2`. `doc_3` (the three hypothetical documents), joined with `". "` |
| `FL`  | `label_text` lines joined with `" "`, then `subject_terms` joined with `", "` and appended. **Raw `snc_codes` are excluded** — they're filing-system tokens (e.g. `"POL 15-1"`), not query vocabulary, and `BM25Model.search()`'s alphanumeric-only cleaning would mangle the `-` anyway. |
| `AUG` | `DOC` text + `FL` text, joined with `". "` |

The final query sent to each model is `f"{original_query}. {augmentation_text}"`
(original query first, so term position/frequency still favors it), built in
`RunGenerator.get_augmented_query()` and applied in `produce_topics_results()`.

## Known limitation: ColBERT query truncation

ColBERT's query encoder (pylate / ColBERTv2) truncates queries to a fixed
length (commonly ~32 tokens). The `DOC` and `AUG` variants add several hundred
words of augmentation text per query, which likely gets mostly or entirely
truncated away for any config that includes ColBERT (`V`, `X`, `C`, `Z`, `W`,
and their hybrid variants — CONFIGS offsets 1, 2, 9, 10, 12, 13, 14, 15).

**Decision**: run these configs anyway rather than excluding them. A run where
ColBERT's queries end up truncated to something close to the un-augmented
query is still a valid (if likely close-to-null) data point for that model;
excluding them would leave gaps in the config x query-field x variant grid.
When interpreting results, treat ColBERT-containing configs' `DOC`/`AUG` rows
with this caveat in mind — a lack of effect there may reflect truncation, not
a finding about the augmentation itself. `FL` variant text is short enough
(a handful of label lines + subject terms) that it's less likely to be fully
truncated.

## Naming convention

The query-field tag gets a 4th "slot" character that the augmentation variant
overwrites (dropping the tag's last character and appending the variant):

| Query field | Base padded tag | + `DOC` | + `FL` | + `AUG` |
|---|---|---|---|---|
| T   | `T---` | `T--DOC`  | `T--FL`  | `T--AUG`  |
| TD  | `TD--` | `TD-DOC`  | `TD-FL`  | `TD-AUG`  |
| TDN | `TDN-` | `TDNDOC`  | `TDNFL`  | `TDNAUG`  |

Run folder names follow the existing `run_new_experiments.py` pattern:
`U5.<tag>.<config-suffix>`, e.g. `U5.T--DOC.V.TOFS.mx--2.b`,
`U5.TD-FL.B.TOFS.-.-`, `U5.TDNAUG.W.TOFS.-.c`.

## Scope

- All 15 `CONFIGS` from `run_new_experiments.py` (offsets 1–15) — same
  configs, unmodified, just with an augmented query.
- All 3 query fields: T, TD, TDN.
- All 3 augmentation variants: DOC, FL, AUG.
- 15 × 3 × 3 = **135 experiments**.
- `U5` only (default `docs_per_box=5`, uniform sampling) — the `DOCSBOX`
  sweep (U1–U4/K5/A-) and the `OFFICIAL` ECF re-runs are **not** repeated for
  augmented queries.
- Standard `random` protocol, 30 seeds (`RANDOM_SEED_LIST`), same as
  `run_standard`/`run_hybrid` in `run_new_experiments.py`.
- Hybrid configs (offsets 1, 2, 14, 15) apply the *same* augmented query to
  both the main ranker and the ALLFL partner ranker used in the RRF fusion.

## Running

From the project root (not from `/src` — this script lives under the
`llm_experiments` package and is run as a module):

```bash
python -m src.llm_experiments.runs.run_query_augmentated_experiments              # all 135
python -m src.llm_experiments.runs.run_query_augmentated_experiments 1001 1002    # specific ids
python -m src.llm_experiments.runs.run_query_augmentated_experiments T TD         # every config/variant for those query fields
python -m src.llm_experiments.runs.run_query_augmentated_experiments DOC          # every config/query-field for that variant only
```

Output lands in `all_runs/<name>/` (same layout/marker files as
`run_new_experiments.py`); an experiment already completed there is skipped
automatically (`is_experiment_done`).

## Experiment ID scheme

`id = 1000 + query_field_offset + variant_offset + config_offset`

- `query_field_offset`: T=0, TD=100, TDN=200
- `variant_offset`: DOC=0, FL=15, AUG=30
- `config_offset`: 1–15 (matches `CONFIGS` offsets in `run_new_experiments.py`)

Range: 1001–1245.

## Run-tracking table

Fill in dates/status as batches complete. One row per (query field, variant);
each row covers 15 configs (30 seeds each) = 450 seed-runs.

| Query field | Variant | ID range  | Status | Notes |
|---|---|---|---|---|
| T   | DOC | 1001–1015 | not started | |
| T   | FL  | 1016–1030 | not started | |
| T   | AUG | 1031–1045 | not started | |
| TD  | DOC | 1101–1115 | not started | |
| TD  | FL  | 1116–1130 | not started | |
| TD  | AUG | 1131–1145 | not started | |
| TDN | DOC | 1201–1215 | not started | |
| TDN | FL  | 1216–1230 | not started | |
| TDN | AUG | 1231–1245 | not started | |
