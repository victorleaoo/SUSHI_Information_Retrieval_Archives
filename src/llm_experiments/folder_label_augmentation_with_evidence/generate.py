"""
Generates a folder-context augmentation for every folder in FoldersV1.3.json, grounded in
documentary evidence sampled from a given seed's ECF training set: a folder's own training
documents when it has any this seed, otherwise documents borrowed from the nearest related
folders -- same SNC, then similar SNC, then same box (create_folder_relations_for_expansion
in src/run_generator.py) -- capped at 5 documents, randomly sampled when a pool is larger.
Folders with no evidence at all in a given seed's sample fall back to
folder_label_augmentation's classification-only prompt.

Usage (from the project root, so it survives a terminal disconnect):
    nohup python -m src.llm_experiments.folder_label_augmentation_with_evidence.generate \
        > data/llm_calls/folder_label_augmentation_with_evidence/generate.out 2>&1 &

Writes one file per seed to
data/llm_calls/folder_label_augmentation_with_evidence/2_folder_context/seed_{seed}.json,
a dict keyed by folder ID with the same shape as folder_label_augmentation's output, plus
`evidence_source` ("own" / "same_snc" / "similar_snc" / "same_box" / "none") and
`evidence_docnos`. Like folder_label_augmentation.generate, each seed's file is re-read on
startup and only folders whose stored result is already valid are skipped, so it's safe to
re-run at any time to backfill missing folders and repair invalid ones.
"""

import json
import os
import random
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

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
from src.llm_experiments.folder_label_augmentation.generate import (
    build_folder_fields,
    load_folders,
    normalize_field,
)
from src.llm_experiments.folder_label_augmentation_with_evidence.prompts import (
    render_folder_context_with_evidence_prompt,
    render_own_evidence_block,
    render_related_evidence_block,
)

# run_generator.py uses bare `from models import ...` imports and a cwd-relative
# `pd.read_excel("RGdistribution.xlsx")` call, so it can only be imported with src/ on
# sys.path and as the working directory (the same way run_generator.py itself expects
# to be run).
sys.path.insert(0, SRC_DIR)
os.chdir(SRC_DIR)
from run_generator import RANDOM_SEED_LIST, RunGenerator  # noqa: E402

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "data", "llm_calls", "folder_label_augmentation_with_evidence", "2_folder_context"
)
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "llm_calls", "cache")
LOG_PATH = os.path.join(PROJECT_ROOT, "data", "llm_calls", "calls.jsonl")

MAX_PARSE_ATTEMPTS = 3
MAX_EVIDENCE_DOCS = 5
CALL_SLEEP_SECONDS = 5

# Order in which relation pools are tried once a folder has no training documents of its
# own this seed; stops at the first non-empty pool (never merges across relation types).
RELATION_FALLBACK_SEQUENCE = [
    ("same_snc", "same snc"),
    ("similar_snc", "similar snc"),
    ("same_box", "same box"),
]


def log(message: str):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def build_training_set(gen: RunGenerator, ecf: dict) -> list:
    """Minimal per-doc records (docno/folder/box/date) needed by
    create_folder_relations_for_expansion -- mirrors the doc-level loop in
    RunGenerator.prepare_training_data, without the text_blob building that's irrelevant
    here."""
    training_set = []
    for training_doc in ecf["ExperimentSets"][0]["TrainingDocuments"]:
        file = training_doc[-10:-4]
        item = gen.items[file]
        training_set.append(
            {
                "docno": file,
                "folder": item["Sushi Folder"],
                "box": item["Sushi Box"],
                "date": item["date"],
            }
        )
    return training_set


def doc_title_summary(gen: RunGenerator, docno: str):
    item = gen.items[docno]
    title = normalize_field(item.get("title"))
    summary = normalize_field(item.get("summary"))
    if not title and not summary:
        return None
    return " — ".join(part for part in (title, summary) if part)


def select_evidence(gen: RunGenerator, relations: dict, folder_id: str, seed: int):
    """Returns (evidence_source, docs), where docs is a list of
    {"docno", "relation", "text"} dicts, capped at MAX_EVIDENCE_DOCS and deterministically
    sampled when a pool is larger. evidence_source is "none" with an empty list when the
    folder's own documents and every relation pool are empty (or yield no usable
    title/summary text)."""
    rng = random.Random(f"{seed}:{folder_id}")

    def build_docs(pool, relation):
        selected = pool if len(pool) <= MAX_EVIDENCE_DOCS else rng.sample(pool, MAX_EVIDENCE_DOCS)
        docs = [{"docno": d, "relation": relation, "text": doc_title_summary(gen, d)} for d in selected]
        return [d for d in docs if d["text"]]

    own_docs = relations[folder_id]["same folder"]
    if own_docs:
        docs = build_docs(own_docs, "own")
        if docs:
            return "own", docs

    for source_name, relation_key in RELATION_FALLBACK_SEQUENCE:
        pool = relations[folder_id][relation_key]
        if pool:
            docs = build_docs(pool, source_name)
            if docs:
                return source_name, docs

    return "none", []


def build_prompt(fields: dict, evidence_source: str, evidence_docs: list) -> str:
    if evidence_source == "none":
        return render_folder_context_prompt(COLLECTION_CONTEXT, **fields)

    if evidence_source == "own":
        evidence_block = render_own_evidence_block([d["text"] for d in evidence_docs])
    else:
        evidence_block = render_related_evidence_block([(d["relation"], d["text"]) for d in evidence_docs])

    return render_folder_context_with_evidence_prompt(COLLECTION_CONTEXT, evidence_block=evidence_block, **fields)


def load_or_init_output(output_path: str, seed: int) -> dict:
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
        "folders": {},
    }


def save_output(output_path: str, data: dict):
    """Writes atomically (temp file + os.replace) so a kill mid-write can never leave a
    truncated/corrupt seed file behind -- on-disk state is always either the previous
    complete save or the new one, never a partial one. Without this, a kill during
    json.dump would corrupt the *whole* seed file, losing every folder already generated
    for that seed, not just the one in flight."""
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)


def generate_with_retries(runner: LLMRunner, prompt: str, force_first: bool = False) -> tuple:
    """Same retry contract as folder_label_augmentation.generate.generate_with_retries,
    plus a CALL_SLEEP_SECONDS throttle after every live (non-cached) call."""
    entry = None
    parsed = None
    for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
        system_prompt = "" if attempt == 1 else RETRY_SYSTEM_PROMPT
        force = force_first if attempt == 1 else True
        entry = runner.run_with_meta(prompt, system_prompt=system_prompt, force=force)
        if not entry.get("cached"):
            time.sleep(CALL_SLEEP_SECONDS)

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


def generate_for_seed(gen: RunGenerator, seed: int, folders: dict, runner: LLMRunner) -> dict:
    """Generates (or resumes) one seed's folder-context file. On every call -- including
    after a prior run was killed mid-seed -- the seed's file is re-read from disk and, per
    folder, only a missing or previously-invalid entry triggers a (re)generation; a folder
    already marked valid is skipped without any LLM call. Returns a counts dict
    (already_valid / generated_valid / generated_invalid) for the seed-level summary."""
    output_path = os.path.join(OUTPUT_DIR, f"seed_{seed}.json")
    data = load_or_init_output(output_path, seed)

    log(f"=== seed={seed} | building ECF and relations ===")
    ecf = gen.loader.create_random_ecf(seed, sampling="uniform", docs_per_box=5)
    training_set = build_training_set(gen, ecf)
    relations = gen.create_folder_relations_for_expansion(training_set)

    folder_ids = list(folders.keys())
    already_valid_count = sum(
        1 for fid in folder_ids if (data["folders"].get(fid) or {}).get("valid")
    )
    log(
        f"=== seed={seed} | {len(folder_ids)} folders total | "
        f"{already_valid_count} already valid (resuming), {len(folder_ids) - already_valid_count} to (re)generate ==="
    )

    counts = {"already_valid": 0, "generated_valid": 0, "generated_invalid": 0}

    for i, folder_id in enumerate(folder_ids, start=1):
        existing = data["folders"].get(folder_id)
        if existing is not None and existing.get("valid"):
            counts["already_valid"] += 1
            log(f"[seed {seed}] ({i}/{len(folder_ids)}) {folder_id} already valid, skipping")
            continue

        folder = folders[folder_id]
        fields = build_folder_fields(folder)
        folder_start = time.monotonic()

        evidence_source, evidence_docs = select_evidence(gen, relations, folder_id, seed)
        prompt = build_prompt(fields, evidence_source, evidence_docs)

        redo = existing is not None
        log(
            f"[seed {seed}] ({i}/{len(folder_ids)}) {folder_id} starting "
            f"({'redo, previously invalid' if redo else 'new'}) | evidence={evidence_source} "
            f"({len(evidence_docs)} docs) | snc={fields['snc']} | label={fields['folder_label']}"
        )

        entry, parsed, attempts_used, valid = generate_with_retries(runner, prompt, force_first=redo)

        data["folders"][folder_id] = {
            "folder_id": folder_id,
            "fields": fields,
            "evidence_source": evidence_source,
            "evidence_docnos": [d["docno"] for d in evidence_docs],
            "raw_response": entry.get("response", ""),
            "parsed": parsed,
            "valid": valid,
            "attempts_used": attempts_used,
            "input_tokens": entry.get("input_tokens"),
            "output_tokens": entry.get("output_tokens"),
            "duration_seconds": entry.get("duration_seconds"),
            "created_at": entry.get("created_at"),
        }

        save_output(output_path, data)
        folder_duration = time.monotonic() - folder_start
        counts["generated_valid" if valid else "generated_invalid"] += 1
        log(
            f"[seed {seed}] ({i}/{len(folder_ids)}) {folder_id} done in {folder_duration:.2f}s "
            f"(valid={valid}, attempts={attempts_used}, evidence={evidence_source}) -> saved to {output_path}"
        )

    log(
        f"=== seed={seed} finished | already_valid={counts['already_valid']} "
        f"generated_valid={counts['generated_valid']} generated_invalid={counts['generated_invalid']} "
        f"(total={len(folder_ids)}) -> {output_path} ==="
    )
    return counts


def main(seeds=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runner = LLMRunner(cache_dir=CACHE_DIR, log_path=LOG_PATH)
    folders = load_folders()
    seeds = seeds if seeds is not None else RANDOM_SEED_LIST

    log(
        f"Starting folder_label_augmentation_with_evidence generation | model={runner.model} | "
        f"seeds={len(seeds)} | output_dir={OUTPUT_DIR}"
    )
    log(
        "Resumable: if killed, re-running this command re-reads each seed's file and only "
        "(re)generates folders that are missing or previously invalid; already-valid folders "
        "are skipped with no LLM call."
    )

    # Loader/items/folderMetadata are seed-independent; build RunGenerator once and reuse
    # it across seeds, only regenerating the ECF and relation graph per seed.
    gen = RunGenerator()

    totals = {"already_valid": 0, "generated_valid": 0, "generated_invalid": 0}
    run_start = time.monotonic()

    for seed in seeds:
        seed_counts = generate_for_seed(gen, seed, folders, runner)
        for key in totals:
            totals[key] += seed_counts[key]

    run_duration = time.monotonic() - run_start
    log(
        f"Done in {run_duration:.2f}s | seeds={len(seeds)} | "
        f"already_valid={totals['already_valid']} generated_valid={totals['generated_valid']} "
        f"generated_invalid={totals['generated_invalid']} "
        f"(total={totals['already_valid'] + totals['generated_valid'] + totals['generated_invalid']})"
    )


if __name__ == "__main__":
    main()
