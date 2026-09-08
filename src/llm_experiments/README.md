# LLM Experiments

Code map for `src/llm_experiments/`. Updated as new prompt sets are added.

## Shared files

- **`llm_runner.py`**
  - `LLMRunner` class — calls the UnB IA-Router (OpenAI-compatible LiteLLM proxy).
  - Caches every prompt on disk (SHA1 hash of the prompt text) — same prompt is never billed twice.
  - Retries a failed call up to 3 times, 10 seconds apart; returns `""` if all 3 fail.
  - Logs every call (cache hit or live) as one line in a JSONL file.
  - `run(prompt)` -> just the response text.
  - `run_with_meta(prompt)` -> full record: response text, input/output token counts, raw API response, `created_at`, `duration_seconds`.
- **`collection_context.py`**
  - `COLLECTION_CONTEXT` — the paragraph of historical/archival background (State Dept. Subject-Numeric Files, Brazil 1963-1973) that gets prepended to most prompts, so it's written once and reused everywhere.
- **`output_schema.py`**
  - `OUTPUT_SCHEMA` — a reusable "CORE_THEMES / RELATED_CONCEPTS" output-format block, with `{{N_WORDS}}`/`{{N_TERMS}}` placeholders.
  - `render_output_schema(n_words, n_terms)` — fills those placeholders.

## `query_expansion/doc_folder_hip/`

Experiment #1: query expansion via hypothetical documents and hypothetical folder labels (HyDE-style), for the 45 SUSHI topics.

- **`prompts.py`**
  - The two prompt templates (documents / folder label), copied from `query_llm_call.md`.
  - `render_documents_prompt(context, query_text)` / `render_folder_label_prompt(context, query_text)` — fill in the template placeholders.
- **`parsing.py`**
  - `parse_documents_response(raw_text)` -> `{"doc_1": ..., "doc_2": ..., "doc_3": ...}`
  - `parse_folder_label_response(raw_text)` -> `{"snc_codes": [...], "label_text": [...], "subject_terms": [...]}`
- **`generate.py`** — the runnable batch script.
  - For each query type — `T` (Title only), `TD` (Title+Description), `TDN` (Title+Description+Narrative) — loops over all 45 topics.
  - For each topic, calls **both** prompts (documents + folder label).
  - Writes results to `data/llm_calls/query_expansion/1_doc_folder_hip/{T,TD,TDN}.json`.
  - Safe to stop and restart: it skips topics already saved in the output file, and `LLMRunner`'s cache means a rerun never repeats an API call.
  - Run it with:
    ```bash
    nohup python -m src.llm_experiments.query_expansion.doc_folder_hip.generate \
        > data/llm_calls/query_expansion/generate.out 2>&1 &
    ```

### Example: one topic entry inside `data/llm_calls/query_expansion/1_doc_folder_hip/TD.json`

```json
{
  "query_type": "TD",
  "model": "UnB-Llama-3.3-70B-Instruct",
  "generated_at": "2026-09-04T22:10:00.123456",
  "topics": {
    "T18Eval-00002": {
      "topic_id": "T18Eval-00002",
      "original_query": {
        "TITLE": "Brazilian submarines",
        "DESCRIPTION": "Find documents that mention the operation of submarines by the Brazilian Navy."
      },
      "query_text": "Brazilian submarines\nFind documents that mention the operation of submarines by the Brazilian Navy.",

      "documents": {
        "raw_response": "DOC_1: This document is about the Brazilian Navy's submarine fleet based at...\nDOC_2: This document is about...\nDOC_3: This document is about...",
        "parsed": {
          "doc_1": "This document is about the Brazilian Navy's submarine fleet based at...",
          "doc_2": "This document is about...",
          "doc_3": "This document is about..."
        },
        "input_tokens": 245,
        "output_tokens": 165,
        "duration_seconds": 1.84,
        "created_at": "2026-09-04T22:10:00.123456"
      },

      "folder_label": {
        "raw_response": "SNC_CODES: DEF 19-8, POL 15-1\nLABEL_TEXT: DEF 19-8 Naval Forces & Equipment BRAZ 1969-1973\nPOL 15-1 Political Parties & Groups BRAZ 1969-1973\nSUBJECT_TERMS: Brazilian Navy, submarines, naval procurement, Marinha do Brasil",
        "parsed": {
          "snc_codes": ["DEF 19-8", "POL 15-1"],
          "label_text": [
            "DEF 19-8 Naval Forces & Equipment BRAZ 1969-1973",
            "POL 15-1 Political Parties & Groups BRAZ 1969-1973"
          ],
          "subject_terms": ["Brazilian Navy", "submarines", "naval procurement", "Marinha do Brasil"]
        },
        "input_tokens": 260,
        "output_tokens": 130,
        "duration_seconds": 1.51,
        "created_at": "2026-09-04T22:10:01.987654"
      }
    }
  }
}
```

Every other topic in the file follows the exact same shape, just keyed by its own `topic_id`.

## `folder_label_augmentation/`

Experiment #2: for every archival folder (all 1,336 in `FoldersV1.3.json`), describes what
its (undigitized) contents would plausibly concern, based only on its Subject-Numeric
classification metadata (label, SNC code, expanded meaning, scope note, date range).

- **`prompts.py`**
  - `FOLDER_CONTEXT_PROMPT_TEMPLATE` — the classification-only prompt, using the shared `OUTPUT_SCHEMA` (`N_WORDS=100-150`, `N_TERMS=15-20`).
  - `render_folder_context_prompt(collection_context, folder_label, snc, parent_expanded_snc, scope_note, start_date, end_date)` — missing fields are passed as `None` and rendered as the literal text `"None"` in the prompt.
  - `RETRY_SYSTEM_PROMPT` — a system prompt used only on retries (see below), so a retry doesn't just replay the same cached response.
- **`parsing.py`**
  - `parse_folder_context_response(raw_text)` -> `{"core_themes": "...", "related_concepts": [...]}`.
  - `is_valid_parsed(parsed)` — `False` if `core_themes` is null/empty/under 25 characters, or `related_concepts` is empty.
- **`generate.py`** — the runnable batch script.
  - Loops over all 1,336 folders in `data/folders_metadata/FoldersV1.3.json`.
  - Normalizes missing metadata: empty strings and the literal string `"nan"` (a pandas artifact) both become `None`. `label1963` is never used for this prompt.
  - `scope_note` = `raw_scope`, falling back to `scope_truncated` if `raw_scope` is empty.
  - For each folder, calls the prompt **up to 3 times**, stopping as soon as `is_valid_parsed()` passes; attempts after the first use `RETRY_SYSTEM_PROMPT` so they bypass the cache and ask the model to fill in what was missing. If all 3 attempts stay invalid, the last (invalid) result is kept and flagged.
  - Writes everything to **one file**: `data/llm_calls/folder_label_augmentation/2_folder_context/folders.json`, a dict keyed by folder ID.
  - Safe to stop and restart: skips folder IDs already saved; `LLMRunner`'s cache means a rerun never repeats an identical API call.
  - Logs progress verbosely (timestamped, `(i/1336)` counters, per-attempt validity/tokens/duration) so a long `nohup` run is easy to follow.
  - Run it with:
    ```bash
    nohup python -m src.llm_experiments.folder_label_augmentation.generate \
        > data/llm_calls/folder_label_augmentation/generate.out 2>&1 &
    ```

### Example: one folder entry inside `data/llm_calls/folder_label_augmentation/2_folder_context/folders.json`

```json
{
  "generated_at": "2026-09-05T00:09:39.916281",
  "folders": {
    "A99990001": {
      "folder_id": "A99990001",
      "fields": {
        "folder_label": "LAB 3 Organizations & Conferences 1964 (Classified)",
        "snc": "LAB 3",
        "parent_expanded_snc": "LABOR & MANPOWER: ORGANIZATIONS & CONFERENCES",
        "scope_note": null,
        "start_date": "01/01/1964",
        "end_date": "01/01/1969"
      },
      "raw_response": "CORE_THEMES: This folder contains records concerning labor organizations and conferences in Brazil...\nRELATED_CONCEPTS: Ministerio do Trabalho, ILO, CLT, labor unions, ...",
      "parsed": {
        "core_themes": "This folder contains records concerning labor organizations and conferences in Brazil...",
        "related_concepts": ["Ministerio do Trabalho", "ILO", "CLT", "labor unions"]
      },
      "valid": true,
      "attempts_used": 1,
      "input_tokens": 310,
      "output_tokens": 190,
      "duration_seconds": 2.05,
      "created_at": "2026-09-05T00:09:39.000000"
    }
  }
}
```

## `runs/`

Consumes the LLM outputs above to drive actual IR experiments (not more LLM
calls). See `runs/runs.md` for full details and a run-tracking table.

- **`run_query_augmentated_experiments.py`** — re-runs the 15 core `CONFIGS`
  from `src/run_new_experiments.py`, for T/TD/TDN, against three query
  variants built from `query_expansion/doc_folder_hip`'s output (`DOC` =
  hypothetical documents, `FL` = hypothetical folder label, `AUG` = both).
  135 experiments total. Depends on `RunGenerator`'s `query_augmentation`
  param in `src/run_generator.py`.

## Where things get stored

- `data/llm_calls/cache/` — `LLMRunner`'s raw per-prompt cache (one JSON file per SHA1 hash of the prompt). This is the low-level cache; you normally don't need to look in here.
- `data/llm_calls/calls.jsonl` — append-only log of every single call made, across all experiments (useful for auditing/cost tracking).
- `data/llm_calls/query_expansion/1_doc_folder_hip/{T,TD,TDN}.json` — structured, per-topic results for Experiment #1 (see example above).
- `data/llm_calls/folder_label_augmentation/2_folder_context/folders.json` — structured, per-folder results for Experiment #2 (see example above).
