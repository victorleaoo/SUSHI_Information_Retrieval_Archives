"""
build_seed_summaries.py
-----------------------
Writes seed_ndcg5.json into every run folder that has Random*_TopicsFolderMetrics.json files: the per-topic
nDCG@5 of each seed, the only part of those files the web app reads. With it, a copy of the runs can leave
the per-seed files out (1.25 GB over all runs; the summaries are ~22 KB per run).

    cd web_app && python3 build_seed_summaries.py           # write / refresh every summary
    cd web_app && python3 build_seed_summaries.py --check   # verify summaries against the seed files

Values are written with json's float repr, so they read back bit-identical (seed-independence compares them
exactly).
"""

import argparse
import json
import os
import sys

import numpy as np

import utils_runs as ur


def summary_of(path):
    seeds = {os.path.basename(f).split("_")[0]: ur._topic_vector(f) for f in ur._seed_files(path)}
    return {"metric": "ndcg_cut_5", "topics": ur.TOPIC_KEYS,
            "seeds": {s: [None if np.isnan(x) else float(x) for x in v] for s, v in seeds.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare existing summaries with the seed files")
    args = ap.parse_args()

    runs = ur.scan_runs()
    written = checked = bad = skipped = 0
    for name, info in sorted(runs.items()):
        if not ur._seed_files(info.path):
            skipped += 1                       # AllFolderLabel / OfficialECF runs, or a copy without seed files
            continue
        target = os.path.join(info.path, ur.SEED_SUMMARY)
        summary = summary_of(info.path)
        if args.check:
            checked += 1
            try:
                with open(target, encoding="utf-8") as f:
                    if json.load(f) != summary:
                        bad += 1
                        print(f"STALE {name}")
            except (OSError, ValueError):
                bad += 1
                print(f"MISSING {name}")
            continue
        with open(target, "w", encoding="utf-8") as f:
            json.dump(summary, f, separators=(",", ":"))
        written += 1
    if args.check:
        print(f"{checked} summaries checked, {bad} missing or stale; {skipped} runs without seed files")
        sys.exit(1 if bad else 0)
    print(f"{written} summaries written; {skipped} runs without seed files (nothing to summarize)")


if __name__ == "__main__":
    main()
