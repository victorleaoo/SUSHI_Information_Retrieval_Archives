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
  - Writes results to `data/llm_calls/query_expansion/T.json`, `TD.json`, `TDN.json`.
  - Safe to stop and restart: it skips topics already saved in the output file, and `LLMRunner`'s cache means a rerun never repeats an API call.
  - Run it with:
    ```bash
    nohup python -m src.llm_experiments.query_expansion.doc_folder_hip.generate \
        > data/llm_calls/query_expansion/generate.out 2>&1 &
    ```

### Example: one topic entry inside `data/llm_calls/query_expansion/TD.json`

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

## Where things get stored

- `data/llm_calls/cache/` — `LLMRunner`'s raw per-prompt cache (one JSON file per SHA1 hash of the prompt). This is the low-level cache; you normally don't need to look in here.
- `data/llm_calls/calls.jsonl` — append-only log of every single call made, across all experiments (useful for auditing/cost tracking).
- `data/llm_calls/query_expansion/{T,TD,TDN}.json` — the structured, human-readable results for the `doc_folder_hip` experiment (see example above). This is what you'll actually use for analysis.
