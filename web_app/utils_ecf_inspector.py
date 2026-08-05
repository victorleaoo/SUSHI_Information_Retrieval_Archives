"""
utils_ecf_inspector.py
----------------------
Helpers for the ECF Inspector page.
Lists available ECFs, parses training documents, computes coverage metrics,
and cross-references with qrels for relevance coverage analysis.
"""

import os
import json
import re
from collections import defaultdict
from pathlib import Path

import streamlit as st
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_ECF_DIR = _ROOT / "ecf" / "random_generated"
_ECF_ALL_PATH = _ECF_DIR / "ECF_ALL_TRAINING_SET.json"
_QRELS_FOLDERS_PATH = _ROOT / "qrels" / "formal-folder-qrel.txt"
_FOLDERS_PATH = _ROOT / "data" / "folders_metadata" / "FoldersV1.3.json"
_ITEMS_PATH = _ROOT / "data" / "items_metadata" / "itemsV1.2.json"



# ──────────────────────────────────────────────────────────────────────
# ECF Discovery
# ──────────────────────────────────────────────────────────────────────

def list_available_ecfs() -> list:
    """
    Scan ecf/random_generated/ and return a list of dicts:
    [{"filename", "label", "path", "type", "seed"}]
    type: "Uniform", "Skewed", "All Docs"
    """
    ecfs = []
    if not _ECF_DIR.exists():
        return ecfs

    for f in sorted(_ECF_DIR.iterdir()):
        if not f.name.endswith(".json"):
            continue

        if f.name.startswith("ECF_ALL"):
            ecf_type = "All Docs"
            seed = "N/A"
            label = "All Docs (31,681 documents)"
        elif f.name.startswith("ECF_UNEVEN"):
            ecf_type = "Skewed"
            match = re.search(r'Seed[_]?(\d+)', f.name)
            seed = match.group(1) if match else "?"
            label = f"Skewed — Seed {seed}"
        elif f.name.startswith("ECF_RANDOM"):
            ecf_type = "Uniform"
            match = re.search(r'RANDOM[_]?(\d+)', f.name)
            seed = match.group(1) if match else "?"
            label = f"Uniform — Seed {seed}"
        else:
            continue

        ecfs.append({
            "filename": f.name,
            "label": label,
            "path": str(f),
            "type": ecf_type,
            "seed": seed,
        })

    return ecfs


# ──────────────────────────────────────────────────────────────────────
# ECF Loading & Parsing
# ──────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading ECF data…")
def load_ecf_training_docs(path: str) -> list:
    """Returns the TrainingDocuments list from an ECF JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for es in data.get("ExperimentSets", []):
            if "TrainingDocuments" in es:
                return es["TrainingDocuments"]
    except Exception:
        pass
    return []


@st.cache_data(show_spinner="Loading folder metadata…")
def _load_folders_meta() -> dict:
    with open(_FOLDERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading item metadata…")
def _load_items_meta() -> dict:
    with open(_ITEMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading folder qrels…")
def _load_folder_qrels() -> dict:
    """topic_id -> [(folder_id, grade), ...] with grade > 0 only."""
    qrels = defaultdict(list)
    try:
        with open(_QRELS_FOLDERS_PATH, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 4:
                    topic_id = parts[0]
                    folder_id = parts[2]
                    try:
                        grade = int(parts[3])
                    except ValueError:
                        continue
                    if grade > 0:
                        qrels[topic_id].append((folder_id, grade))
    except FileNotFoundError:
        pass
    return dict(qrels)


@st.cache_data(show_spinner="Loading topic titles…")
def _load_topic_titles() -> dict:
    titles = {}
    if _ECF_ALL_PATH.exists():
        try:
            with open(_ECF_ALL_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for es in data.get("ExperimentSets", []):
                for tid, tinfo in es.get("Topics", {}).items():
                    if isinstance(tinfo, dict) and "TITLE" in tinfo:
                        titles[tid] = tinfo["TITLE"]
        except Exception:
            pass
    return titles


def parse_ecf_docs(training_docs: list) -> dict:
    """
    Parse ECF TrainingDocuments paths (Box/Folder/File.pdf) into structured data.
    Returns:
        doc_ids: set of document IDs (filename without .pdf)
        folder_ids: set of folder IDs that have ≥1 doc
        box_ids: set of box IDs
        folder_doc_counts: {folder_id: count}
        folder_doc_ids: {folder_id: [doc_id, ...]}
        box_folder_counts: {box_id: set of folder_ids with docs}
        box_doc_counts: {box_id: total doc count}
    """
    doc_ids = set()
    folder_ids = set()
    box_ids = set()
    folder_doc_counts = defaultdict(int)
    folder_doc_ids = defaultdict(list)
    box_folder_set = defaultdict(set)
    box_doc_counts = defaultdict(int)

    for path_str in training_docs:
        parts = path_str.split("/")
        if len(parts) >= 3:
            box_id = parts[0]
            folder_id = parts[1]
            doc_file = parts[2]
            doc_id = doc_file[:-4] if doc_file.lower().endswith(".pdf") else doc_file

            doc_ids.add(doc_id)
            folder_ids.add(folder_id)
            box_ids.add(box_id)
            folder_doc_counts[folder_id] += 1
            folder_doc_ids[folder_id].append(doc_id)
            box_folder_set[box_id].add(folder_id)
            box_doc_counts[box_id] += 1

    return {
        "doc_ids": doc_ids,
        "folder_ids": folder_ids,
        "box_ids": box_ids,
        "folder_doc_counts": dict(folder_doc_counts),
        "folder_doc_ids": dict(folder_doc_ids),
        "box_folder_set": {k: v for k, v in box_folder_set.items()},
        "box_doc_counts": dict(box_doc_counts),
    }


# ──────────────────────────────────────────────────────────────────────
# Coverage Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_snc_coverage(ecf_data: dict, folders_meta: dict) -> pd.DataFrame:
    """
    For each SNC code, compute total folders, covered folders, coverage %.
    Returns a DataFrame sorted by coverage % ascending.
    """
    covered_folders = ecf_data["folder_ids"]

    snc_total = defaultdict(int)
    snc_covered = defaultdict(int)

    for fid, meta in folders_meta.items():
        snc = meta.get("snc", "Unknown")
        primary = snc.split()[0] if snc else "Unknown"
        snc_total[primary] += 1
        if fid in covered_folders:
            snc_covered[primary] += 1

    rows = []
    for snc in sorted(snc_total.keys()):
        total = snc_total[snc]
        covered = snc_covered.get(snc, 0)
        rows.append({
            "SNC": snc,
            "Total Folders": total,
            "Covered Folders": covered,
            "Uncovered Folders": total - covered,
            "Coverage %": round(covered / total * 100, 1) if total > 0 else 0.0,
        })

    return pd.DataFrame(rows).sort_values("Coverage %", ascending=True).reset_index(drop=True)


def compute_snc_3level_coverage(ecf_data: dict, folders_meta: dict) -> pd.DataFrame:
    """Same as compute_snc_coverage but at 3-level SNC granularity."""
    covered_folders = ecf_data["folder_ids"]

    snc_total = defaultdict(int)
    snc_covered = defaultdict(int)

    for fid, meta in folders_meta.items():
        snc = meta.get("snc", "Unknown").strip()
        snc_total[snc] += 1
        if fid in covered_folders:
            snc_covered[snc] += 1

    rows = []
    for snc in sorted(snc_total.keys()):
        total = snc_total[snc]
        covered = snc_covered.get(snc, 0)
        rows.append({
            "SNC": snc,
            "Total Folders": total,
            "Covered Folders": covered,
            "Uncovered Folders": total - covered,
            "Coverage %": round(covered / total * 100, 1) if total > 0 else 0.0,
        })

    return pd.DataFrame(rows).sort_values("Coverage %", ascending=True).reset_index(drop=True)


def compute_box_coverage(ecf_data: dict, folders_meta: dict) -> pd.DataFrame:
    """
    For each box, compute total folders, covered folders, doc count, coverage %.
    """
    covered_folders = ecf_data["folder_ids"]
    box_doc_counts = ecf_data.get("box_doc_counts", {})

    box_total = defaultdict(int)
    box_covered = defaultdict(int)

    for fid, meta in folders_meta.items():
        box = meta.get("box", "Unknown")
        box_total[box] += 1
        if fid in covered_folders:
            box_covered[box] += 1

    rows = []
    for box in sorted(box_total.keys()):
        total = box_total[box]
        covered = box_covered.get(box, 0)
        rows.append({
            "Box": box,
            "Folders Total": total,
            "Folders Covered": covered,
            "Doc Count": box_doc_counts.get(box, 0),
            "Coverage %": round(covered / total * 100, 1) if total > 0 else 0.0,
        })

    return pd.DataFrame(rows).sort_values("Coverage %", ascending=True).reset_index(drop=True)


def compute_folder_detail(ecf_data: dict, folders_meta: dict) -> pd.DataFrame:
    """
    Returns a DataFrame of folders covered by the ECF (i.e., folders that have
    at least one training document in this ECF), with SNC, label, box, and doc count.
    Dropdown label format: "FOLDER_ID — SNC — Label"
    """
    covered_folder_doc_counts = ecf_data.get("folder_doc_counts", {})

    rows = []
    for fid, doc_count in covered_folder_doc_counts.items():
        meta = folders_meta.get(fid, {})
        snc = meta.get("snc", "Unknown")
        label = meta.get("label", "N/A")
        box = meta.get("box", "N/A")
        date = meta.get("date", "N/A")
        end_date = meta.get("endDate", "N/A")
        scope = meta.get("raw_scope", "")
        label_parent_expanded = meta.get("label_parent_expanded", "")
        rows.append({
            "Folder ID": fid,
            "SNC": snc,
            "Label": label,
            "Box": box,
            "Date": date,
            "End Date": end_date,
            "Scope Note": scope if scope and str(scope) != "nan" else "",
            "Label Parent Expanded": label_parent_expanded,
            "Docs in Training Set": doc_count,
            "Dropdown Label": f"{fid} — {snc} — {label}",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["SNC", "Folder ID"]).reset_index(drop=True)
    return df


def compute_relevance_coverage(ecf_data: dict) -> dict:
    """
    Cross-reference ECF coverage with qrels.
    Returns:
        summary: {total_pairs, covered_pairs, uncovered_pairs,
                  total_grade3, covered_grade3, total_grade1, covered_grade1}
        per_topic: DataFrame with Topic, # Relevant Folders, # Covered, # Uncovered, Coverage %
    """
    qrels = _load_folder_qrels()
    topic_titles = _load_topic_titles()
    covered_folders = ecf_data["folder_ids"]

    total_pairs = 0
    covered_pairs = 0
    total_g3 = 0
    covered_g3 = 0
    total_g1 = 0
    covered_g1 = 0

    topic_rows = []

    for topic_id in sorted(qrels.keys()):
        rel_folders = qrels[topic_id]
        n_rel = len(rel_folders)
        n_covered = sum(1 for fid, _ in rel_folders if fid in covered_folders)
        n_uncovered = n_rel - n_covered

        total_pairs += n_rel
        covered_pairs += n_covered

        for fid, grade in rel_folders:
            if grade == 3:
                total_g3 += 1
                if fid in covered_folders:
                    covered_g3 += 1
            elif grade == 1:
                total_g1 += 1
                if fid in covered_folders:
                    covered_g1 += 1

        match = re.search(r'\d+$', str(topic_id))
        num = match.group() if match else topic_id
        title = topic_titles.get(topic_id, "")
        topic_label = f"T{num} — {title}" if title else f"T{num}"

        topic_rows.append({
            "Topic": topic_label,
            "Relevant Folders": n_rel,
            "Covered": n_covered,
            "Uncovered": n_uncovered,
            "Coverage %": round(n_covered / n_rel * 100, 1) if n_rel > 0 else 0.0,
        })

    return {
        "summary": {
            "total_pairs": total_pairs,
            "covered_pairs": covered_pairs,
            "uncovered_pairs": total_pairs - covered_pairs,
            "total_grade3": total_g3,
            "covered_grade3": covered_g3,
            "total_grade1": total_g1,
            "covered_grade1": covered_g1,
        },
        "per_topic": pd.DataFrame(topic_rows),
    }


def compute_headline_metrics(ecf_data: dict, folders_meta: dict) -> dict:
    """Compute top-level metrics for the ECF Inspector headline row."""
    total_docs = len(ecf_data["doc_ids"])
    total_items = len(_load_items_meta())
    covered_folders = len(ecf_data["folder_ids"])
    total_folders = len(folders_meta)
    covered_boxes = len(ecf_data["box_ids"])

    all_boxes = set(m.get("box", "") for m in folders_meta.values())
    total_boxes = len(all_boxes)

    avg_docs_per_folder = (
        total_docs / covered_folders if covered_folders > 0 else 0.0
    )

    return {
        "docs_included": total_docs,
        "docs_total": total_items,
        "docs_pct": round(total_docs / total_items * 100, 1) if total_items > 0 else 0,
        "folders_covered": covered_folders,
        "folders_total": total_folders,
        "folders_pct": round(covered_folders / total_folders * 100, 1) if total_folders > 0 else 0,
        "boxes_covered": covered_boxes,
        "boxes_total": total_boxes,
        "boxes_pct": round(covered_boxes / total_boxes * 100, 1) if total_boxes > 0 else 0,
        "avg_docs_per_folder": round(avg_docs_per_folder, 2),
    }
