"""Step 2 (and Step 6): experiment 3, one description per folder from its classification.

Writes <out>/folder_label_augmentation/2_folder_context/folders.json in the old shape.

Usage (project root):
  nohup python -m src.dspy_experiments.generate_folder_side --out hw \
      > data/llm_calls_luna/logs/folders_hw.out 2>&1 &
"""
import argparse
import collections
import json
import os
from datetime import datetime

import dspy

from src.dspy_experiments import lm as _lm  # noqa: F401
from src.dspy_experiments.common import (LUNA_ROOT, MODEL_NAME, build_folder_fields, folder_inputs,
                                         load_folders, load_json, log, save_json, usage_of)
from src.dspy_experiments.signatures import DescribeFolder, ok_folder, refined, run_batch


def folder_record(folder_id, fields, pred, valid, program, extra=None, attempts=1):
    tin, tout = usage_of(pred) if pred is not None else (None, None)
    rec = {
        "folder_id": folder_id,
        "fields": fields,
        "raw_response": json.dumps(dict(pred), ensure_ascii=False) if pred is not None else "",
        "parsed": ({"core_themes": pred.core_themes, "related_concepts": list(pred.related_concepts or [])}
                   if pred is not None else {"core_themes": None, "related_concepts": []}),
        "valid": valid,
        "attempts": attempts,
        "program": program,
        "input_tokens": tin,
        "output_tokens": tout,
        "created_at": datetime.now().isoformat(),
    }
    rec.update(extra or {})
    return rec


def attempts_summary(folders_data):
    """Histogram of rounds needed per folder: {attempts: count}, plus how many never passed."""
    hist = collections.Counter(v["attempts"] for v in folders_data.values() if v["valid"])
    stuck = sum(1 for v in folders_data.values() if not v["valid"])
    parts = ", ".join(f"{n} in {k} attempt{'s' if k != 1 else ''}" for k, n in sorted(hist.items()))
    return f"{parts}{', ' if parts else ''}{stuck} never valid" if stuck else parts


def generate(out_root, program="hw", instruction=None, threads=10, repair_rounds=1, chunk=200):
    path = os.path.join(out_root, "folder_label_augmentation", "2_folder_context", "folders.json")
    data = load_json(path, {"model": MODEL_NAME, "generated_at": datetime.now().isoformat(), "folders": {}})
    folders = load_folders()
    signature = DescribeFolder.with_instructions(instruction) if instruction else DescribeFolder
    module = refined(signature, ok_folder)

    for round_ in range(repair_rounds + 1):
        todo = [fid for fid in folders if not data["folders"].get(fid, {}).get("valid")]
        log(f"round {round_}: {len(todo)} folders to (re)generate")
        for start in range(0, len(todo), chunk):            # save after every chunk
            ids = todo[start:start + chunk]
            fields = {fid: build_folder_fields(folders[fid]) for fid in ids}
            results = run_batch(module, ok_folder, [folder_inputs(fields[fid]) for fid in ids],
                                rollout_start=10 * round_, num_threads=threads)
            for fid, (pred, valid) in zip(ids, results):
                attempts = data["folders"].get(fid, {}).get("attempts", 0) + 1
                data["folders"][fid] = folder_record(fid, fields[fid], pred, valid, program, attempts=attempts)
            save_json(path, data)
            log(f"  saved {start + len(ids)}/{len(todo)}")
    log(f"{sum(v['valid'] for v in data['folders'].values())}/{len(folders)} folders valid "
        f"({attempts_summary(data['folders'])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hw")
    ap.add_argument("--instruction-file", default=None, help="optimized docstring (Step 6)")
    ap.add_argument("--program", default="hw")
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    instruction = open(args.instruction_file).read() if args.instruction_file else None
    out_root = os.path.join(LUNA_ROOT, args.out)
    with dspy.track_usage() as usage:
        generate(out_root, args.program, instruction, args.threads)
    save_json(os.path.join(out_root, "usage", "folder_side.json"), usage.get_total_tokens())


if __name__ == "__main__":
    main()