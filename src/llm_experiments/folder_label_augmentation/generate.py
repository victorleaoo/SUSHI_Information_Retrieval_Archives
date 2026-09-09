"""
Generates a folder-context augmentation for every folder in FoldersV1.3.json,
describing what an undigitized folder's contents would plausibly concern,
based purely on its Subject-Numeric classification metadata.

Usage (from the project root, so it survives a terminal disconnect):
    nohup python -m src.llm_experiments.folder_label_augmentation.generate \
        > data/llm_calls/folder_label_augmentation/generate.out 2>&1 &

Writes a single file, data/llm_calls/folder_label_augmentation/2_folder_context/folders.json,
a dict keyed by folder ID (the FoldersV1.3.json key, e.g. "A99990001"). Each entry holds
the classification fields used, the raw response, the parsed fields (core_themes,
related_concepts), whether the parse was valid, how many attempts it took, and token/
timing metadata. The script re-reads its own output file on startup and skips only folders
whose stored result is already valid; a folder that's missing entirely or previously came
out invalid is (re)generated, forcing past any stale cache entry so a re-run actually fixes
it rather than replaying the same bad response. It's therefore safe and useful to re-run
this script at any time to backfill missing folders and repair invalid ones, at no cost for
folders that are already valid.

Each folder is retried up to 3 times if the parsed result comes out invalid (core_themes
null/empty/<25 chars, or related_concepts empty) -- each retry uses a distinguishing
system prompt so it doesn't just replay the same cached (invalid) response.
"""

import json
import os
import time
from datetime import datetime

from src.llm_experiments.collection_context import COLLECTION_CONTEXT
from src.llm_experiments.llm_runner import LLMRunner
from src.llm_experiments.folder_label_augmentation.parsing import (
    is_valid_parsed,
    parse_folder_context_response,
)
from src.llm_experiments.folder_label_augmentation.prompts import (
    RETRY_SYSTEM_PROMPT,
    render_folder_context_prompt,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FOLDERS_PATH = os.path.join(PROJECT_ROOT, "data", "folders_metadata", "FoldersV1.3.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "llm_calls", "folder_label_augmentation", "2_folder_context")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "folders.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "llm_calls", "cache")
LOG_PATH = os.path.join(PROJECT_ROOT, "data", "llm_calls", "calls.jsonl")

MAX_PARSE_ATTEMPTS = 3


def load_folders() -> dict:
    with open(FOLDERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_field(value):
    """Missing/blank/pandas-NaN-as-string values all collapse to None."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return text


def build_folder_fields(folder: dict) -> dict:
    scope_note = normalize_field(folder.get("raw_scope")) or normalize_field(folder.get("scope_truncated"))
    return {
        "folder_label": normalize_field(folder.get("label")),
        "snc": normalize_field(folder.get("snc")),
        "parent_expanded_snc": normalize_field(folder.get("label_parent_expanded")),
        "scope_note": scope_note,
        "start_date": normalize_field(folder.get("date")),
        "end_date": normalize_field(folder.get("endDate")),
    }


def load_or_init_output() -> dict:
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "generated_at": datetime.now().isoformat(),
        "folders": {},
    }


def save_output(data: dict):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log(message: str):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def generate_with_retries(runner: LLMRunner, prompt: str, force_first: bool = False) -> tuple:
    """Calls the prompt up to MAX_PARSE_ATTEMPTS times until parsed output is valid.
    Returns (entry, parsed, attempts_used, valid).

    Attempts 2+ all share one cache key (same RETRY_SYSTEM_PROMPT + prompt), so without
    forcing, attempt 3 would just replay attempt 2's cached response instead of making a
    real third call -- force=True for attempt > 1 guarantees each retry is a live call.
    `force_first` additionally bypasses the cache on attempt 1, for redoing a folder that
    a previous run already gave up on as invalid (replaying that stale response would just
    reproduce the same invalid result).
    """
    entry = None
    parsed = None
    for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
        system_prompt = "" if attempt == 1 else RETRY_SYSTEM_PROMPT
        force = force_first if attempt == 1 else True
        entry = runner.run_with_meta(prompt, system_prompt=system_prompt, force=force)
        parsed = parse_folder_context_response(entry.get("response", ""))
        valid = is_valid_parsed(parsed)

        status = "CACHED" if entry.get("cached") else "LIVE"
        duration = entry.get("duration_seconds")
        duration_str = f"{duration:.2f}s" if duration is not None else "n/a"
        log(
            f"        attempt {attempt}/{MAX_PARSE_ATTEMPTS} {status} | "
            f"valid={valid} | in={entry.get('input_tokens')} out={entry.get('output_tokens')} tokens | {duration_str}"
        )

        if valid:
            return entry, parsed, attempt, True

    log(f"        giving up after {MAX_PARSE_ATTEMPTS} attempts, keeping last (invalid) result")
    return entry, parsed, MAX_PARSE_ATTEMPTS, False


def generate_for_all_folders(runner: LLMRunner):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_or_init_output()
    folders = load_folders()
    folder_ids = list(folders.keys())

    log(f"=== folder_context | {len(folder_ids)} folders total ===")

    for i, folder_id in enumerate(folder_ids, start=1):
        existing = data["folders"].get(folder_id)
        if existing is not None and existing.get("valid"):
            log(f"({i}/{len(folder_ids)}) {folder_id} already valid, skipping")
            continue

        folder = folders[folder_id]
        fields = build_folder_fields(folder)
        folder_start = time.monotonic()

        redo = existing is not None
        log(
            f"({i}/{len(folder_ids)}) {folder_id} starting ({'redo, previously invalid' if redo else 'new'}) "
            f"| snc={fields['snc']} | label={fields['folder_label']}"
        )

        prompt = render_folder_context_prompt(COLLECTION_CONTEXT, **fields)
        entry, parsed, attempts_used, valid = generate_with_retries(runner, prompt, force_first=redo)

        data["folders"][folder_id] = {
            "folder_id": folder_id,
            "fields": fields,
            "raw_response": entry.get("response", ""),
            "parsed": parsed,
            "valid": valid,
            "attempts_used": attempts_used,
            "input_tokens": entry.get("input_tokens"),
            "output_tokens": entry.get("output_tokens"),
            "duration_seconds": entry.get("duration_seconds"),
            "created_at": entry.get("created_at"),
        }

        save_output(data)
        folder_duration = time.monotonic() - folder_start
        log(f"({i}/{len(folder_ids)}) {folder_id} done in {folder_duration:.2f}s (valid={valid}, attempts={attempts_used}) -> saved to {OUTPUT_PATH}")

    log("=== folder_context finished ===")


def main():
    runner = LLMRunner(cache_dir=CACHE_DIR, log_path=LOG_PATH)
    log(f"Starting folder_label_augmentation generation | model={runner.model} | output={OUTPUT_PATH}")
    generate_for_all_folders(runner)
    log("Done.")


if __name__ == "__main__":
    main()
