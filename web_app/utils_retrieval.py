"""
utils_retrieval.py
------------------
Helpers for the Retrieval Analysis section of the Experiment Analyzer.
Loads run.txt rankings, folder-level qrels, and LLM augmentation data
for a given run directory.
"""

import os
import json
from collections import defaultdict

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_ROOT, "data")
_QRELS_FOLDERS_PATH = os.path.join(_ROOT, "qrels", "formal-folder-qrel.txt")


@st.cache_data
def load_folder_qrels() -> dict:
    """Returns: topic_id -> {folder_id: grade}"""
    qrels: dict = defaultdict(dict)
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
                    qrels[topic_id][folder_id] = grade
    except FileNotFoundError:
        pass
    return dict(qrels)


@st.cache_data
def load_run_rankings(run_dir: str, top_n: int = 15) -> dict:
    """
    Reads run.txt from run_dir.
    Format: topic_id  folder_id  rank  [score  ...]
    Returns: topic_id -> [(rank, folder_id), ...] sorted ascending, capped at top_n.
    """
    run_file = os.path.join(run_dir, "run.txt")
    topic_ranking: dict = defaultdict(list)
    if os.path.exists(run_file):
        with open(run_file, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 3:
                    topic_id = parts[0]
                    folder_id = parts[1]
                    try:
                        rank = int(parts[2])
                    except ValueError:
                        continue
                    if rank <= top_n:
                        topic_ranking[topic_id].append((rank, folder_id))
    for tid in topic_ranking:
        topic_ranking[tid].sort(key=lambda x: x[0])
    return dict(topic_ranking)


@st.cache_data
def load_layer23_for_run(run_dir: str) -> dict:
    """Attempts to find the layer23 augmentation JSON matching the seed in run_dir."""
    for seed in [1, 42, 100, 300, 333]:
        if f"seed{seed}" in run_dir or f"_seed{seed}" in run_dir:
            path = os.path.join(_DATA_DIR, f"layer23_augmentation_ECF_RANDOM_{seed}.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    return json.load(fh)
    return {}


def grade_badge(grade: int) -> str:
    if grade == 3:
        return "🟢 Highly Relevant (3)"
    elif grade == 1:
        return "🟡 Relevant (1)"
    elif grade == 0:
        return "⚫ Not Relevant (0)"
    return f"Grade {grade}"


def grade_stars(grade: int) -> str:
    """Star-based grade display matching Topic Viewer convention."""
    if grade == 3:
        return "⭐⭐⭐"
    elif grade == 1:
        return "⭐"
    return "—"


def ndcg_at_k(ranking: list, qrels_topic: dict, k: int = 5) -> float:
    import math
    grades = sorted(qrels_topic.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))
    if idcg == 0:
        return 0.0
    dcg = sum((2 ** qrels_topic.get(fid, 0) - 1) / math.log2(i + 2) for i, (_, fid) in enumerate(ranking[:k]))
    return dcg / idcg


def compute_all_topics_summary(run_dir: str, folder_qrels: dict,
                                top_n_values: list = None) -> 'pd.DataFrame':
    """
    Build a summary DataFrame across all topics for a given run.
    Columns: Topic, nDCG@5, Rel in Top-5, Rel in Top-15, Rel in Top-25, Rel in Top-50.
    """
    import pandas as pd
    if top_n_values is None:
        top_n_values = [5, 10, 15, 20]

    max_n = max(top_n_values)
    rankings = load_run_rankings(run_dir, top_n=max_n)
    if not rankings:
        return pd.DataFrame()

    rows = []
    for topic_id in sorted(rankings.keys()):
        ranking = rankings[topic_id]
        qrels = folder_qrels.get(topic_id, {})
        ndcg5 = ndcg_at_k(ranking, qrels, k=5)
        row = {"Topic": topic_id, "nDCG@5": round(ndcg5, 4)}
        for n in top_n_values:
            rel_count = sum(1 for _, fid in ranking[:n] if qrels.get(fid, 0) > 0)
            row[f"Rel Top-{n}"] = rel_count
        rows.append(row)

    return pd.DataFrame(rows)

