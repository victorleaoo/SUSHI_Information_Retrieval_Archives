#!/usr/bin/env python3
"""
all_runs_rename_dotted.py
=========================
Renames experiment folders in `all_runs/` from the old encoded naming convention
to the new 5-element dotted notation:

    Sample.Ranker.Fields.LabelSearch.ScorePropagation

And flattens the directory structure so every run sits directly under `all_runs/`.

Usage:
    python scripts/all_runs_rename_dotted.py               # Dry run (preview)
    python scripts/all_runs_rename_dotted.py --apply        # Apply changes

Naming Convention (5-Element Dotted Notation):
──────────────────────────────────────────────
Position 1 — Sample:
    U = Uniform 630 docs (5 per box)
    K = sKewed 630 docs
    A = All 31,681 docs

Position 2 — Ranker:
    B = Unweighted BM25F
    C = ColBERT
    E = Embedding Similarity (all-mpnet-base-v2)
    X = RRF(B + C)
    Y = RRF(B + E)
    Z = RRF(B + C + E)
    W = RRF(Weighted BM25 + C + E)

Position 3 — Fields (4 chars, positions T-O-F-S):
    T = Title, O = OCR, F = Folder Label, S = Summary
    Dashes fill unused positions. Examples: T---, -O--, TOFS, T-FS

Position 4 — Label Search:
    L = weighted RRF with full-collection folder Label search
    x = no label search

Position 5 — Score Propagation:
    1 = same SNC code depth 1
    2 = same SNC code depth 2
    x = no score propagation

Example: U.B.T-FS.L.2 = Uniform, BM25, Title+Folder+Summary, Label search, propagation depth 2
"""

import os
import shutil
import argparse
from typing import Dict, List, Tuple, Optional

# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ALL_RUNS_DIR = os.path.join(PROJECT_ROOT, "all_runs")

# ──────────────────────────────────────────
# EXPLICIT RENAME MAPPING: old_basename -> new_dotted_name
# ──────────────────────────────────────────

RENAME_MAPPING: Dict[str, str] = {
    # ── Category 1: Fields = T--- (Title only) ──
    "1_A_BM25":             "U.B.T---.x.x",
    "1_A_CBERT":            "U.C.T---.x.x",
    "1_A_EMB":              "U.E.T---.x.x",
    "1_A_BM25-CBERT":       "U.X.T---.x.x",
    "1_A_BM25-EMB":         "U.Y.T---.x.x",
    "1_A_BM25-EMB-CBERT":   "U.Z.T---.x.x",

    # ── Category 2: Fields = --F- (Folder Label only) ──
    "2_A_BM25":             "U.B.--F-.x.x",
    "2_A_CBERT":            "U.C.--F-.x.x",
    "2_A_EMB":              "U.E.--F-.x.x",
    "2_A_BM25-CBERT":       "U.X.--F-.x.x",
    "2_A_BM25-EMB":         "U.Y.--F-.x.x",
    "2_A_BM25-EMB-CBERT":   "U.Z.--F-.x.x",

    # ── Category 3: Fields = -O-- (OCR only) ──
    "3_A_BM25":             "U.B.-O--.x.x",
    "3_A_CBERT":            "U.C.-O--.x.x",
    "3_A_EMB":              "U.E.-O--.x.x",
    "3_A_BM25-CBERT":       "U.X.-O--.x.x",
    "3_A_BM25-EMB":         "U.Y.-O--.x.x",
    "3_A_BM25-EMB-CBERT":   "U.Z.-O--.x.x",

    # ── Category 4: Fields = ---S (Summary only) ──
    "4_A_BM25":             "U.B.---S.x.x",
    "4_A_CBERT":            "U.C.---S.x.x",
    "4_A_EMB":              "U.E.---S.x.x",
    "4_A_BM25-CBERT":       "U.X.---S.x.x",
    "4_A_BM25-EMB":         "U.Y.---S.x.x",
    "4_A_BM25-EMB-CBERT":   "U.Z.---S.x.x",

    # ── Category 5: Fields = T-FS (Title + Folder + Summary) ──
    "5_A_BM25":                         "U.B.T-FS.x.x",
    "5_A_CBERT":                        "U.C.T-FS.x.x",
    "5_A_EMB":                          "U.E.T-FS.x.x",
    "5_A_BM25-CBERT":                   "U.X.T-FS.x.x",
    "5_A_BM25-EMB":                     "U.Y.T-FS.x.x",
    "5_A_BM25-EMB-CBERT":               "U.Z.T-FS.x.x",
    "5_A_BM25(TUNED)-EMB-CBERT":        "U.W.T-FS.x.x",

    # ── Category 6: Fields = TOF- (Title + OCR + Folder) ──
    "6_A_BM25":             "U.B.TOF-.x.x",
    "6_A_CBERT":            "U.C.TOF-.x.x",
    "6_A_EMB":              "U.E.TOF-.x.x",
    "6_A_BM25-CBERT":       "U.X.TOF-.x.x",
    "6_A_BM25-EMB":         "U.Y.TOF-.x.x",
    "6_A_BM25-EMB-CBERT":   "U.Z.TOF-.x.x",

    # ── Category 7: Fields = TOFS (All fields) ──
    "7_A_BM25":                                     "U.B.TOFS.x.x",
    "7_A_CBERT":                                    "U.C.TOFS.x.x",
    "7_A_EMB":                                      "U.E.TOFS.x.x",
    "7_A_BM25-CBERT":                               "U.X.TOFS.x.x",
    "7_A_BM25-EMB":                                 "U.Y.TOFS.x.x",
    "7_A_BM25-EMB-CBERT":                           "U.Z.TOFS.x.x",
    "7_A_BM25(TUNED)-EMB-CBERT":                    "U.W.TOFS.x.x",
    # Score propagation variants:
    "7_CBERT-D-2_BM25(TUNED)-EMB-CBERT":            "U.C.TOFS.x.2",
    "7_D-1_BM25(TUNED)-EMB-CBERT":                  "U.W.TOFS.x.1",
    "7_D-2_BM25(TUNED)-EMB-CBERT":                  "U.W.TOFS.x.2",

    # ── Category 8: Fields = ---- (No doc fields, Label search only) ──
    "8_A_BM25":             "U.B.----.L.x",
    "8_A_CBERT":            "U.C.----.L.x",
    "8_A_EMB":              "U.E.----.L.x",
    "8_A_BM25-CBERT":       "U.X.----.L.x",
    "8_A_BM25-EMB":         "U.Y.----.L.x",
    "8_A_BM25-EMB-CBERT":   "U.Z.----.L.x",

    # ── UNEVEN (sKewed sampling) ──
    "UNEVEN-7_A_BM25-EMB-CBERT":           "K.Z.TOFS.x.x",
    "UNEVEN-7_A_BM25(TUNED)-EMB-CBERT":    "K.W.TOFS.x.x",

    # ── ALLDOCS (All 31,681 docs) ──
    "ALLDOCS_9_A_BM25-EMB":                "A.Y.T-FS.x.x",

    # ── HYBRID (Label search + Document search, WRRF 0.85 only) ──
    "HYBRID-7-A-8-CBERT_NE_BM25(TUNED)-EMB-CBERT-WRRF085":         "U.W.TOFS.L.x",
    "HYBRID-7-BM25-2-8-CBERT_NE_BM25(TUNED)-EMB-CBERT-WRRF085":    "U.W.TOFS.L.2",
}

# ──────────────────────────────────────────
# RUNS TO DELETE (unclear mapping or superseded WRRF variants)
# ──────────────────────────────────────────

RUNS_TO_DELETE: List[str] = [
    # Unclear expansion variants in category 7
    "7_BM25-1_BM25(TUNED)-EMB-CBERT",
    "7_BM25-2_BM25(TUNED)-EMB-CBERT",
    "7_BM25-3_BM25(TUNED)-EMB-CBERT",
    "7_CBERT-2_BM25(TUNED)-EMB-CBERT",
    "7_CBERT-BM25-2_BM25(TUNED)-EMB-CBERT",
    "7_D-BM25-2_BM25(TUNED)-EMB-CBERT",
    "7_EMB-2_BM25-EMB-CBERT",
    # Unclear expansion variant in category 5
    "5_BM25-2_BM25(TUNED)-EMB-CBERT",
    # HYBRID runs that are superseded (keep only WRRF085)
    "HYBRID-7-A-8-CBERT_NE_BM25(TUNED)-EMB-CBERT",
    "HYBRID-7-A-8-CBERT_NE_BM25(TUNED)-EMB-CBERT-WRRF",
    "HYBRID-7-A-8-CBERT_NE_BM25(TUNED)-EMB-CBERT-WRRF05",
    "HYBRID-7-A-8-CBERT_NE_BM25(TUNED)-EMB-CBERT-WRRF065",
    "HYBRID-7-A-8-CBERT_NE_BM25(TUNED)-EMB-CBERT-WRRF075",
    "HYBRID-7-A-8-CBERT_NE_BM25(TUNED)-EMB-CBERT-WRRFEQ",
    "HYBRID-7-A-8-CBERT_NE_BM25(TUNED)-EMB-WRRF065",
    "HYBRID-7-BM25-1-8-CBERT_NE_BM25(TUNED)-EMB-CBERT-WRRF",
    "HYBRID-7-BM25-2-8-CBERT_NE_BM25(TUNED)-EMB-CBERT-WRRF",
]


# ==========================================
# CORE FUNCTIONS
# ==========================================

def find_all_run_folders(all_runs_dir: str) -> List[Tuple[str, str]]:
    """
    Recursively find all run folders (containing metric JSON files).
    Returns list of (full_path, basename) tuples.
    """
    valid_metric_files = [
        'topics_mean_margin.json',
        'model_overall_stats.json',
        'all_documents_model_overall_stats.json',
        'AllDocuments_TopicsFolderMetrics.json',
        'topics_values.json',
        'folder_overall_stats.json',
    ]
    results = []
    for root, dirs, files in os.walk(all_runs_dir):
        dirs.sort()
        if any(f in files for f in valid_metric_files):
            results.append((root, os.path.basename(root)))
    return results


def print_mapping_table():
    """Print the complete mapping table for documentation."""
    print("\n" + "=" * 90)
    print("  COMPLETE RENAME MAPPING TABLE")
    print("=" * 90)
    print(f"  {'Old Name':<55} {'New Dotted Name'}")
    print("-" * 90)

    categories = {
        "Cat 1 (T--- Title only)": [],
        "Cat 2 (--F- Folder only)": [],
        "Cat 3 (-O-- OCR only)": [],
        "Cat 4 (---S Summary only)": [],
        "Cat 5 (T-FS Title+Folder+Summary)": [],
        "Cat 6 (TOF- Title+OCR+Folder)": [],
        "Cat 7 (TOFS All fields)": [],
        "Cat 8 (---- Label search only)": [],
        "UNEVEN (sKewed)": [],
        "ALLDOCS (All docs)": [],
        "HYBRID (Label+Doc search)": [],
    }

    for old, new in RENAME_MAPPING.items():
        if old.startswith("1_"):     categories["Cat 1 (T--- Title only)"].append((old, new))
        elif old.startswith("2_"):   categories["Cat 2 (--F- Folder only)"].append((old, new))
        elif old.startswith("3_"):   categories["Cat 3 (-O-- OCR only)"].append((old, new))
        elif old.startswith("4_"):   categories["Cat 4 (---S Summary only)"].append((old, new))
        elif old.startswith("5_"):   categories["Cat 5 (T-FS Title+Folder+Summary)"].append((old, new))
        elif old.startswith("6_"):   categories["Cat 6 (TOF- Title+OCR+Folder)"].append((old, new))
        elif old.startswith("7_"):   categories["Cat 7 (TOFS All fields)"].append((old, new))
        elif old.startswith("8_"):   categories["Cat 8 (---- Label search only)"].append((old, new))
        elif old.startswith("UNEVEN"):  categories["UNEVEN (sKewed)"].append((old, new))
        elif old.startswith("ALLDOCS"): categories["ALLDOCS (All docs)"].append((old, new))
        elif old.startswith("HYBRID"):  categories["HYBRID (Label+Doc search)"].append((old, new))

    for cat_name, entries in categories.items():
        if entries:
            print(f"\n  -- {cat_name} --")
            for old, new in entries:
                print(f"  {old:<55} -> {new}")

    print(f"\n  Total mappings: {len(RENAME_MAPPING)}")
    print(f"  Total runs to delete: {len(RUNS_TO_DELETE)}")
    print("=" * 90)


def execute_rename(all_runs_dir: str, apply: bool = False):
    """
    Main execution: find all run folders, rename/flatten/delete them.
    """
    run_folders = find_all_run_folders(all_runs_dir)

    if not run_folders:
        print("No run folders found in", all_runs_dir)
        return

    print(f"\nFound {len(run_folders)} run folders in {all_runs_dir}")

    mode = "APPLYING CHANGES" if apply else "DRY RUN PREVIEW"
    print(f"\n{'=' * 70}")
    print(f"  {mode}")
    print(f"{'=' * 70}")

    used_names: Dict[str, int] = {}
    renamed_count = 0
    deleted_count = 0
    unknown_runs = []

    # Phase 1: Process renames and deletions
    for full_path, basename in sorted(run_folders, key=lambda x: x[1]):
        rel_path = os.path.relpath(full_path, all_runs_dir)

        # Check if it should be deleted
        if basename in RUNS_TO_DELETE:
            if apply:
                shutil.rmtree(full_path)
                print(f"  [DELETED]  {rel_path}")
            else:
                print(f"  [DELETE]   {rel_path}")
            deleted_count += 1
            continue

        # Check if it has a mapping
        if basename not in RENAME_MAPPING:
            unknown_runs.append((full_path, rel_path, basename))
            continue

        new_name = RENAME_MAPPING[basename]

        # Handle collisions
        if new_name in used_names:
            used_names[new_name] += 1
            counter = used_names[new_name]
            final_name = f"{new_name}.{counter}"
        else:
            used_names[new_name] = 1
            final_name = new_name

        target_path = os.path.join(all_runs_dir, final_name)

        if apply:
            if os.path.exists(target_path):
                print(f"  [SKIP]     Target already exists: {final_name}")
            else:
                shutil.move(full_path, target_path)
                print(f"  [RENAMED]  {rel_path}")
                print(f"           -> {final_name}")
                renamed_count += 1
        else:
            print(f"  [RENAME]   {rel_path}")
            print(f"           -> {final_name}")
            renamed_count += 1

    # Phase 2: Report unknown runs
    if unknown_runs:
        print(f"\n  WARNING: UNKNOWN RUNS (not in mapping or delete list):")
        for _, rel_path, basename in unknown_runs:
            print(f"    - {rel_path} (basename: {basename})")

    # Phase 3: Cleanup empty directories
    if apply:
        print(f"\n  Cleaning up empty directories...")
        cleanup_count = 0
        for root, dirs, files in os.walk(all_runs_dir, topdown=False):
            if root == all_runs_dir:
                continue
            if not os.listdir(root):
                os.rmdir(root)
                rel = os.path.relpath(root, all_runs_dir)
                print(f"  [REMOVED]  Empty dir: {rel}")
                cleanup_count += 1
        print(f"  Removed {cleanup_count} empty directories.")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    verb = "Renamed" if apply else "Would rename"
    dverb = "Deleted" if apply else "Would delete"
    print(f"  {verb}: {renamed_count} folders")
    print(f"  {dverb}: {deleted_count} folders")
    if unknown_runs:
        print(f"  Unknown/unmapped: {len(unknown_runs)} folders")
    if not apply:
        print(f"\n  This was a DRY RUN. Use '--apply' to perform actual changes.")
    print(f"{'=' * 70}\n")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename all_runs experiments to 5-element dotted notation and flatten structure."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply actual renames/deletes on disk (default is dry-run preview).",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print the complete mapping table and exit.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=ALL_RUNS_DIR,
        help=f"Path to all_runs directory (default: {ALL_RUNS_DIR}).",
    )

    args = parser.parse_args()

    if args.table:
        print_mapping_table()
    else:
        print_mapping_table()
        execute_rename(all_runs_dir=args.dir, apply=args.apply)
