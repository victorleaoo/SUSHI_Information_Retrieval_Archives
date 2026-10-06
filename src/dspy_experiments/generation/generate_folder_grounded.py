"""Step 2 (and Step 6): experiment 4, folder descriptions grounded in each seed's sampled documents.

Evidence rule (unchanged): the folder's own sampled documents, else same SNC, else similar
SNC, else same box; first non-empty pool wins, never mixed; at most 5 documents. Folders with
no evidence use DescribeFolder (identical requests to experiment 3, so the cache answers them).

Writes <out>/folder_label_augmentation_with_evidence/2_folder_context/seed_{seed}.json.

Usage (project root):
  nohup python -m src.dspy_experiments.generate_grounded --out hw \
      > data/llm_calls_luna/logs/grounded_hw.out 2>&1 &
"""
import argparse
import os
import random
import sys
from datetime import datetime

import dspy

from src.dspy_experiments import lm as _lm  # noqa: F401
from src.dspy_experiments.common import (LUNA_ROOT, MODEL_NAME, PROJECT_ROOT, build_folder_fields,
                                         folder_inputs, load_folders, load_json, log,
                                         normalize_field, save_json)
from src.dspy_experiments.generation.generate_folder_side import attempts_summary, folder_record
from src.dspy_experiments.signatures import (DescribeFolder, DescribeFolderWithEvidence, ok_folder,
                                             refined, run_batch)

# run_generator.py uses bare imports and a cwd-relative RGdistribution.xlsx, so import it from src/.
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
os.chdir(SRC_DIR)
from run_generator import RANDOM_SEED_LIST, RunGenerator  # noqa: E402

MAX_EVIDENCE_DOCS = 5
FALLBACK = [("same_snc", "same snc"), ("similar_snc", "similar snc"), ("same_box", "same box")]


def build_training_set(gen, ecf):
    rows = []
    for training_doc in ecf["ExperimentSets"][0]["TrainingDocuments"]:
        docno = training_doc[-10:-4]
        item = gen.items[docno]
        rows.append({"docno": docno, "folder": item["Sushi Folder"],
                     "box": item["Sushi Box"], "date": item["date"]})
    return rows


def doc_text(gen, docno):
    item = gen.items[docno]
    title, summary = normalize_field(item.get("title")), normalize_field(item.get("summary"))
    return " - ".join(p for p in (title, summary) if p) or None


def select_evidence(gen, relations, folder_id, seed):
    """Same rule and same deterministic sampling as the Llama version."""
    rng = random.Random(f"{seed}:{folder_id}")

    def pick(pool):
        chosen = pool if len(pool) <= MAX_EVIDENCE_DOCS else rng.sample(pool, MAX_EVIDENCE_DOCS)
        return [(d, doc_text(gen, d)) for d in chosen if doc_text(gen, d)]

    own = relations[folder_id]["same folder"]
    if own and (docs := pick(own)):
        return "own", docs
    for source, key in FALLBACK:
        pool = relations[folder_id][key]
        if pool and (docs := pick(pool)):
            return source, docs
    return "none", []


def generate_seed(gen, seed, folders, out_root, program, mod_ev, mod_plain, threads, repair_rounds):
    path = os.path.join(out_root, "folder_label_augmentation_with_evidence", "2_folder_context",
                        f"seed_{seed}.json")
    data = load_json(path, {"seed": seed, "model": MODEL_NAME,
                            "generated_at": datetime.now().isoformat(), "folders": {}})
    ecf = gen.loader.create_random_ecf(seed, sampling="uniform", docs_per_box=5)
    relations = gen.create_folder_relations_for_expansion(build_training_set(gen, ecf))
    evidence = {fid: select_evidence(gen, relations, fid, seed) for fid in folders}

    for round_ in range(repair_rounds + 1):
        todo = [fid for fid in folders if not data["folders"].get(fid, {}).get("valid")]
        if not todo:
            break
        log(f"[seed {seed}] round {round_}: {len(todo)} folders")
        for group, module in (("ev", mod_ev), ("plain", mod_plain)):
            ids = [fid for fid in todo if (evidence[fid][0] != "none") == (group == "ev")]
            fields = {fid: build_folder_fields(folders[fid]) for fid in ids}
            inputs = []
            for fid in ids:
                x = folder_inputs(fields[fid])
                if group == "ev":
                    kind, docs = evidence[fid]
                    x.update(evidence_kind=kind, evidence=[text for _, text in docs])
                inputs.append(x)
            results = run_batch(module, ok_folder, inputs, rollout_start=10 * round_, num_threads=threads)
            for fid, (pred, valid) in zip(ids, results):
                kind, docs = evidence[fid]
                attempts = data["folders"].get(fid, {}).get("attempts", 0) + 1
                data["folders"][fid] = folder_record(
                    fid, fields[fid], pred, valid, program, attempts=attempts,
                    extra={"evidence_source": kind, "evidence_docnos": [d for d, _ in docs]})
        save_json(path, data)
        log(f"[seed {seed}] round {round_} done: {attempts_summary(data['folders'])}")
    n = sum(v["valid"] for v in data["folders"].values())
    log(f"[seed {seed}] {n}/{len(folders)} valid -> {path} ({attempts_summary(data['folders'])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hw")
    ap.add_argument("--program", default="hw")
    ap.add_argument("--evidence-instruction-file", default=None, help="transferred docstring (Step 6)")
    ap.add_argument("--plain-instruction-file", default=None, help="experiment 3 docstring (Step 6)")
    ap.add_argument("--seeds", nargs="+", type=int, default=RANDOM_SEED_LIST)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()

    ev_sig, plain_sig = DescribeFolderWithEvidence, DescribeFolder
    if args.evidence_instruction_file:
        ev_sig = ev_sig.with_instructions(open(args.evidence_instruction_file).read())
    if args.plain_instruction_file:
        plain_sig = plain_sig.with_instructions(open(args.plain_instruction_file).read())
    mod_ev, mod_plain = refined(ev_sig, ok_folder), refined(plain_sig, ok_folder)

    out_root = os.path.join(LUNA_ROOT, args.out)
    gen, folders = RunGenerator(), load_folders()
    with dspy.track_usage() as usage:
        for seed in args.seeds:
            generate_seed(gen, seed, folders, out_root, args.program, mod_ev, mod_plain,
                          args.threads, repair_rounds=1)
    save_json(os.path.join(out_root, "usage", "grounded.json"), usage.get_total_tokens())


if __name__ == "__main__":
    main()