import os
import re
import argparse
from typing import Dict, List, Tuple

# ==========================================
# CONFIGURATIONS
# ==========================================

SEARCH_FIELD_CONFIG: Dict[str, int] = {
    "T": 1,
    "F": 2,
    "O": 3,
    "S": 4,
    "TSF": 5,
    "TOF": 6,
    "TOFS": 7,
    "ALLFL": 8,
    "TFS": 9,
}

EXPANSION_CONFIG: Dict[str, str] = {
    "NEX": "A",
    "SMS": "B",
    "SB": "C",
    "SS": "D",
    "CD": "E",
}

MODEL_CONFIG: Dict[str, str] = {
    "BM25-EMBEDDINGS-COLBERT-TUNED": "BM25(TUNED)-EMB-CBERT",
    "BM25-EMBEDDINGS-TUNED": "BM25(TUNED)-EMB",
    "BM25-EMBEDDINGS-COLBERT": "BM25-EMB-CBERT",
    "BM25-EMBEDDINGS": "BM25-EMB",
    "BM25-COLBERT": "BM25-CBERT",
    "COLBERT-TUNED": "CBERT-TUNED",
    "COLBERTTUNED": "CBERT-TUNED",
    "EMBEDDINGS": "EMB",
    "COLBERT": "CBERT",
    "BM25": "BM25",
    "TUNED": "TUNED",
}

# Mapping old shortcodes (B, E, C, T) to new model codes
OLD_MODEL_MAP: Dict[str, str] = {
    "B-E-C-T": "BM25(TUNED)-EMB-CBERT",
    "B-E-T": "BM25(TUNED)-EMB",
    "B-E-C": "BM25-EMB-CBERT",
    "B-E": "BM25-EMB",
    "B-C": "BM25-CBERT",
    "C-T": "CBERT-TUNED",
    "B": "BM25",
    "C": "CBERT",
    "E": "EMB",
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ALL_RUNS_DIR = os.path.join(PROJECT_ROOT, "all_runs")


def remove_td_from_foldername(folder_name: str) -> str:
    """
    Removes 'TD' tags from a run folder name.
    Handles occurrences like '_TD_', '_TD-', '_TD', or 'TD_'.
    """
    new_name = re.sub(r"_TD_", "_", folder_name)
    new_name = re.sub(r"_TD$", "", new_name)
    new_name = re.sub(r"^TD_", "", new_name)
    return new_name


def encode_foldername_with_dicts(folder_name: str) -> str:
    """
    Replaces search field keys (TOFS -> 7, F -> 2), expansion keys (NEX -> A, SMS -> B),
    and model keys (BM25, EMB, CBERT, BM25(TUNED)) in the folder_name.
    """
    new_name = remove_td_from_foldername(folder_name)

    # 1. Replace old model shortcodes (B-E-C-T, B-E, etc.) with new model codes
    sorted_old_model_keys = sorted(OLD_MODEL_MAP.keys(), key=len, reverse=True)
    for key in sorted_old_model_keys:
        val = OLD_MODEL_MAP[key]
        pattern = r"(^|[_\-])" + re.escape(key) + r"(?=$|[_\-])"
        new_name = re.sub(pattern, lambda m, v=val: m.group(1) + v, new_name)

    # 2. Replace search field keys bounded by delimiters (_, -, or start/end)
    sorted_search_keys = sorted(SEARCH_FIELD_CONFIG.keys(), key=len, reverse=True)
    for key in sorted_search_keys:
        val = str(SEARCH_FIELD_CONFIG[key])
        pattern = r"(^|[_\-])" + re.escape(key) + r"(?=$|[_\-])"
        new_name = re.sub(pattern, lambda m, v=val: m.group(1) + v, new_name)

    # 3. Replace expansion keys bounded by delimiters (_, -, or start/end)
    sorted_expansion_keys = sorted(EXPANSION_CONFIG.keys(), key=len, reverse=True)
    for key in sorted_expansion_keys:
        val = EXPANSION_CONFIG[key]
        pattern = r"(^|[_\-])" + re.escape(key) + r"(?=$|[_\-])"
        new_name = re.sub(pattern, lambda m, v=val: m.group(1) + v, new_name)

    # 4. Replace original model names if any remain
    sorted_model_keys = sorted(MODEL_CONFIG.keys(), key=len, reverse=True)
    for key in sorted_model_keys:
        val = MODEL_CONFIG[key]
        pattern = r"(^|[_\-])" + re.escape(key) + r"(?=$|[_\-])"
        new_name = re.sub(pattern, lambda m, v=val: m.group(1) + v, new_name)

    return new_name


def get_folder_rename_mappings(all_runs_dir: str = ALL_RUNS_DIR, use_dicts: bool = False) -> List[Tuple[str, str]]:
    """
    Scans all_runs directory (recursively, bottom-up) and returns a list of (old_path, new_name) tuples for folders needing renaming.
    """
    if not os.path.exists(all_runs_dir):
        print(f"Directory not found: {all_runs_dir}")
        return []

    mappings = []
    for root, dirs, _ in os.walk(all_runs_dir, topdown=False):
        for d in sorted(dirs):
            cleaned_name = encode_foldername_with_dicts(d) if use_dicts else remove_td_from_foldername(d)
            if cleaned_name != d:
                mappings.append((os.path.join(root, d), cleaned_name))
    return mappings


def rename_all_runs(all_runs_dir: str = ALL_RUNS_DIR, apply: bool = False, use_dicts: bool = False) -> None:
    """
    Processes all subdirectories in all_runs.
    If apply is False, previews the changes (dry run).
    If apply is True, renames folders on disk.
    """
    mappings = get_folder_rename_mappings(all_runs_dir, use_dicts=use_dicts)

    if not mappings:
        print("No folders found requiring renaming.")
        return

    mode_str = "APPLYING RENAMES" if apply else "DRY RUN PREVIEW"
    dict_str = " WITH DICTS" if use_dicts else ""
    print(f"\n--- {mode_str}{dict_str} ({len(mappings)} folders) ---")

    renamed_count = 0
    for old_path, new_name in mappings:
        parent_dir = os.path.dirname(old_path)
        old_name = os.path.basename(old_path)
        new_path = os.path.join(parent_dir, new_name)

        if apply:
            if os.path.exists(new_path):
                print(f"[SKIP] Target already exists: {new_name}")
            else:
                os.rename(old_path, new_path)
                print(f"[RENAMED] {old_name}\n       -> {new_name}")
                renamed_count += 1
        else:
            print(f"[WOULD RENAME] {old_name}\n             -> {new_name}")

    if apply:
        print(f"\nSuccessfully renamed {renamed_count} folders.")
    else:
        print(f"\nDry run complete. Use '--apply' to perform the actual folder renames.")


def print_configs() -> None:
    """Prints the current configuration dictionaries."""
    print("Search Field Configuration (SEARCH_FIELD_CONFIG):")
    for key, val in SEARCH_FIELD_CONFIG.items():
        print(f"  - {key}: {val}")

    print("\nExpansion Configuration (EXPANSION_CONFIG):")
    for key, val in EXPANSION_CONFIG.items():
        print(f"  - {key}: '{val}'")

    print("\nModel Configuration (MODEL_CONFIG):")
    for key, val in MODEL_CONFIG.items():
        print(f"  - {key}: '{val}'")


def organize_into_category_folders(all_runs_dir: str = ALL_RUNS_DIR, apply: bool = False) -> None:
    """
    Groups run directories inside all_runs into category subfolders based on the first part of their name (before '_').
    E.g. '1_A_BM25' is moved into 'all_runs/1/1_A_BM25'.
    """
    if not os.path.exists(all_runs_dir):
        print(f"Directory not found: {all_runs_dir}")
        return

    move_operations: List[Tuple[str, str, str]] = []

    for entry in sorted(os.listdir(all_runs_dir)):
        full_path = os.path.join(all_runs_dir, entry)
        if os.path.isdir(full_path) and "_" in entry:
            category = entry.split("_")[0]
            if category and category != entry:
                target_category_dir = os.path.join(all_runs_dir, category)
                target_run_dir = os.path.join(target_category_dir, entry)
                move_operations.append((full_path, category, target_run_dir))

    if not move_operations:
        print("No run folders found to organize into category subfolders.")
        return

    mode_str = "APPLYING CATEGORY ORGANIZATION" if apply else "DRY RUN PREVIEW CATEGORY ORGANIZATION"
    print(f"\n--- {mode_str} ({len(move_operations)} folders) ---")

    moved_count = 0
    for old_path, category, new_path in move_operations:
        old_name = os.path.basename(old_path)
        rel_target = os.path.join(category, old_name)

        if apply:
            target_category_dir = os.path.dirname(new_path)
            os.makedirs(target_category_dir, exist_ok=True)
            if os.path.exists(new_path):
                print(f"[SKIP] Target already exists: {rel_target}")
            else:
                os.rename(old_path, new_path)
                print(f"[MOVED] {old_name} -> {rel_target}")
                moved_count += 1
        else:
            print(f"[WOULD MOVE] {old_name} -> {rel_target}")

    if apply:
        print(f"\nSuccessfully organized {moved_count} run folders into category subfolders.")
    else:
        print(f"\nDry run complete. Use '--organize --apply' to perform the actual folder moves.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean 'all_runs' directory by removing TD, using dict configs, and organizing into category subfolders."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply actual folder renames/moves on disk (defaults to dry-run preview if omitted).",
    )
    parser.add_argument(
        "--use-dicts",
        action="store_true",
        help="Rename search fields to numbers (1..9), expansions to letters (A..E), and models to (BM25, EMB, CBERT, BM25(TUNED)).",
    )
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Organize run folders into category subfolders based on their leading prefix before '_'.",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Display SEARCH_FIELD_CONFIG, EXPANSION_CONFIG, and MODEL_CONFIG dictionaries.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=ALL_RUNS_DIR,
        help="Path to all_runs directory (default: project_root/all_runs).",
    )

    args = parser.parse_args()

    if args.config:
        print_configs()

    if args.organize:
        organize_into_category_folders(all_runs_dir=args.dir, apply=args.apply)
    else:
        rename_all_runs(all_runs_dir=args.dir, apply=args.apply, use_dicts=args.use_dicts)
