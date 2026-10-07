"""Counts, validity and code checks for one generated set.  python -m src.dspy_experiments.check_step2 --out hw"""
import argparse
import glob
import json
import os

from src.dspy_experiments.common import LUNA_ROOT, load_folders, load_topics
from src.dspy_experiments.signatures import MEANING


def share(entries, key=None):
    vals = [(e.get(key) or {}).get("valid", False) if key else e.get("valid", False) for e in entries]
    return f"{sum(vals)}/{len(vals)} valid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hw")
    root = os.path.join(LUNA_ROOT, ap.parse_args().out)
    n_topics, n_folders = len(load_topics()), len(load_folders())

    for qt in ("T", "TD", "TDN"):
        if not os.path.exists(f"{root}/query_expansion/1_doc_folder_hip/{qt}.json"):
            print(f"{qt}: missing")
            continue
        hip = json.load(open(f"{root}/query_expansion/1_doc_folder_hip/{qt}.json"))["topics"]
        ct = json.load(open(f"{root}/query_expansion/2_core_themes_and_related_concepts/{qt}.json"))["topics"]
        bad_codes = [c for e in hip.values() for c in e["folder_label"]["parsed"].get("snc_codes", [])
                     if c not in MEANING]
        print(f"{qt}: {len(hip)}/{n_topics} topics | documents {share(hip.values(), 'documents')} | "
              f"filing {share(hip.values(), 'folder_label')} | core themes {share(ct.values(), 'core_themes')} | "
              f"unknown codes: {len(bad_codes)}")

    folders_path = f"{root}/folder_label_augmentation/2_folder_context/folders.json"
    if os.path.exists(folders_path):
        fo = json.load(open(folders_path))["folders"]
        print(f"experiment 3: {len(fo)}/{n_folders} folders | {share(fo.values())}")

    seeds = sorted(glob.glob(f"{root}/folder_label_augmentation_with_evidence/2_folder_context/seed_*.json"))
    for path in seeds:
        fo = json.load(open(path))["folders"]
        sources = {}
        for e in fo.values():
            sources[e["evidence_source"]] = sources.get(e["evidence_source"], 0) + 1
        print(f"{os.path.basename(path)}: {len(fo)}/{n_folders} | {share(fo.values())} | {sources}")


if __name__ == "__main__":
    main()