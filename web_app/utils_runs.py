"""
utils_runs.py
-------------
The single source of truth for "which runs exist".

Runs live in three roots (nothing is moved or copied):
  all_runs/              legacy runs (6-7 slot names, no LLM)
  all_runs_luna/         the DSPy / GPT-6-luna runs (8-9 slot names: ...TABLE.COLUMN[.QUERY SET])
  all_runs_luna_rerank/  the listwise-reranked luna runs (...RRN)

Only first-level directories that hold a metrics file are runs. `_tmp/`, `order/` (a fix run) and `runs/`
are never entered.
"""

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

ROOTS = {                                   # root key -> directory (relative to PROJECT_ROOT)
    "legacy": "all_runs",
    "luna": "all_runs_luna",
    "luna_rerank": "all_runs_luna_rerank",
}
ROOT_FAMILY = {"legacy": "legacy", "luna": "luna", "luna_rerank": "luna"}
SKIP_DIRS = {"_tmp", "order", "runs"}       # never descend into these

VALID_METRIC_FILES = (
    "topics_mean_margin.json",
    "model_overall_stats.json",
    "all_documents_model_overall_stats.json",
    "AllDocuments_TopicsFolderMetrics.json",
    "topics_values.json",
    "folder_overall_stats.json",
)

QUERY_SET_CODES = ("QCV", "QALL", "QWEB")
RERANKER_CODES = ("RRN", "RRM")
TOPIC_KEYS = [f"T{i}" for i in range(1, 46)]

SEED_INDEPENDENCE_RULE = (
    "Runs whose results do not depend on the training sample (no per-seed files, or identical per-seed files: "
    "ALLFL with fixed folder text, OfficialECF) cannot be paired by seed. The test then pairs the 45 topics, "
    "each run's seed-averaged nDCG@5 per topic."
)


@dataclass(frozen=True)
class RunInfo:
    name: str                 # folder name, unique across roots
    path: str                 # absolute
    family: str               # "legacy" | "luna" (rerank runs are "luna", with the reranker slot set)
    root: str                 # "legacy" | "luna" | "luna_rerank"
    slots: Dict = field(hash=False, compare=False)
    in_sample: bool           # FLOPTALL table or QALL query set
    seed_independent: bool    # see is_seed_independent
    n_seed_files: int         # number of per-seed results (count_seeds)


# ──────────────────────────────────────────────────────────────────────────────
# Name parser
# ──────────────────────────────────────────────────────────────────────────────
def parse_run_name(name: str) -> Dict[str, Optional[str]]:
    """Slots of a run name. Legacy names have 6-7 dotted parts, luna names 8-10."""
    parts = name.split(".")
    slots = {
        "sample": parts[0] if parts else None,
        "qf": parts[1] if len(parts) > 1 else None,
        "config": ".".join(parts[2:6]) if len(parts) >= 6 else None,
        "table": None,
        "column": None,
        "query_set": None,
        "reranker": None,
        "protocol": "30 seeds",
    }
    is_luna = len(parts) >= 8 and not (len(parts) == 7 and parts[6] == "OfficialECF")
    if is_luna:
        slots["table"] = parts[6]
        slots["column"] = parts[7]
        tail = parts[8:]
        slots["query_set"] = next((p for p in tail if p in QUERY_SET_CODES), "hw")
        slots["reranker"] = next((p for p in tail if p in RERANKER_CODES), None)
    elif len(parts) == 7:
        slots["protocol"] = "OfficialECF" if parts[6] == "OfficialECF" else parts[6]
    return slots


def is_in_sample(slots: Dict) -> bool:
    return slots.get("table") == "FLOPTALL" or slots.get("query_set") == "QALL"


# ──────────────────────────────────────────────────────────────────────────────
# Seed independence (D4)
# ──────────────────────────────────────────────────────────────────────────────
def _seed_files(path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(path, "Random*_TopicsFolderMetrics.json")))


def _topic_vector(metrics_file: str) -> np.ndarray:
    """Per-topic nDCG@5 of one seed file, in T1..T45 order (missing topics = nan)."""
    with open(metrics_file, encoding="utf-8") as f:
        data = json.load(f)
    by_topic = {}
    for raw_key, m in data.items():
        digits = "".join(ch for ch in raw_key[-5:] if ch.isdigit())
        if digits and isinstance(m, dict) and "ndcg_cut_5" in m:
            by_topic[f"T{int(digits)}"] = float(m["ndcg_cut_5"])
    return np.array([by_topic.get(t, np.nan) for t in TOPIC_KEYS])


# The per-seed metric files are large (1.25 GB over all runs) and the app reads only their nDCG@5, so a
# run folder may instead hold this one summary file (build_seed_summaries.py writes it):
#   {"metric": "ndcg_cut_5", "topics": ["T1", ..., "T45"], "seeds": {"Random42": [v_T1, ..., v_T45], ...}}
SEED_SUMMARY = "seed_ndcg5.json"


def _read_seed_summary(path: str) -> Dict[str, List]:
    try:
        with open(os.path.join(path, SEED_SUMMARY), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    topics = data.get("topics", TOPIC_KEYS)
    out = {}
    for seed, values in data.get("seeds", {}).items():
        by_topic = dict(zip(topics, values))
        out[seed] = [by_topic.get(t) for t in TOPIC_KEYS]
    return out


def seed_topic_values(path: str) -> Dict[str, np.ndarray]:
    """{seed name ("Random42"): per-topic nDCG@5 in T1..T45 order (missing = nan)}. Read from the
    Random*_TopicsFolderMetrics.json files when present, else from seed_ndcg5.json."""
    files = _seed_files(path)
    if files:
        return {os.path.basename(f).split("_")[0]: _topic_vector(f) for f in files}
    return {seed: np.array([np.nan if v is None else float(v) for v in vals])
            for seed, vals in _read_seed_summary(path).items()}


def count_seeds(path: str, files: Optional[List[str]] = None) -> int:
    """Number of per-seed results: Random*_TopicsFolderMetrics.json files, else seeds in seed_ndcg5.json."""
    files = os.listdir(path) if files is None else files
    n = sum(1 for f in files if f.startswith("Random") and f.endswith("_TopicsFolderMetrics.json"))
    if n == 0 and SEED_SUMMARY in files:
        n = len(_read_seed_summary(path))
    return n


def is_seed_independent_strict(path: str) -> bool:
    """D4, read from the per-seed results themselves: fewer than 2 seeds, or all of them give identical
    per-topic nDCG@5."""
    vecs = list(seed_topic_values(path).values())
    if len(vecs) < 2:
        return True                                   # AllFolderLabel / OfficialECF / single seed
    return all(np.array_equal(vecs[0], v, equal_nan=True) for v in vecs[1:])


def _seed_independent_fast(path: str, n_seed_files: int) -> bool:
    """Same rule as is_seed_independent_strict, read from topics_values.json (one value per seed and topic)
    instead of 30 seed files. Falls back to the seed files when topics_values.json does not line up with them."""
    if n_seed_files < 2:
        return True
    tv_path = os.path.join(path, "topics_values.json")
    try:
        with open(tv_path, encoding="utf-8") as f:
            tv = json.load(f)
        lists = [tv[t] for t in TOPIC_KEYS if t in tv]
        if len(lists) == len(TOPIC_KEYS) and all(isinstance(v, list) and len(v) == n_seed_files for v in lists):
            return all(all(x == v[0] for x in v) for v in lists)
    except (OSError, ValueError, KeyError):
        pass
    return is_seed_independent_strict(path)


def is_seed_independent(path: str) -> bool:
    """D4: a run is seed-independent when it has fewer than 2 Random*.json files, or when all of them give
    identical per-topic nDCG@5. Decided from the files, not the name."""
    for info in scan_runs().values():
        if info.path == path:
            return info.seed_independent
    return _seed_independent_fast(path, count_seeds(path))


# ──────────────────────────────────────────────────────────────────────────────
# Scan
# ──────────────────────────────────────────────────────────────────────────────
def _scan_root(root_key: str) -> List[RunInfo]:
    root_dir = os.path.join(PROJECT_ROOT, ROOTS[root_key])
    if not os.path.isdir(root_dir):
        return []
    out = []
    with os.scandir(root_dir) as it:
        for entry in it:
            if not entry.is_dir() or entry.name in SKIP_DIRS:
                continue
            try:
                files = os.listdir(entry.path)
            except OSError:
                continue
            if not any(f in files for f in VALID_METRIC_FILES):
                continue
            n_seed = count_seeds(entry.path, files)
            slots = parse_run_name(entry.name)
            seed_indep = _seed_independent_fast(entry.path, n_seed)
            if ROOT_FAMILY[root_key] == "luna" and seed_indep:
                slots["protocol"] = "ALLFL"
            out.append(RunInfo(
                name=entry.name,
                path=entry.path,
                family=ROOT_FAMILY[root_key],
                root=root_key,
                slots=slots,
                in_sample=is_in_sample(slots),
                seed_independent=seed_indep,
                n_seed_files=n_seed,
            ))
    return out


@st.cache_data(show_spinner="Scanning runs…")
def scan_runs() -> Dict[str, RunInfo]:
    """{run name: RunInfo} over the three roots. Raises on a duplicate name (none are expected)."""
    runs: Dict[str, RunInfo] = {}
    for root_key in ROOTS:
        for info in _scan_root(root_key):
            if info.name in runs:
                raise RuntimeError(f"Duplicate run name {info.name!r}: {runs[info.name].path} and {info.path}")
            runs[info.name] = info
    return runs


def run_path(name: str) -> Optional[str]:
    info = scan_runs().get(name)
    return info.path if info else None


def run_info(name: str) -> Optional[RunInfo]:
    return scan_runs().get(name)


def load_info_json(name: str) -> Dict:
    """The run's info.json (luna and rerank runs), or {}."""
    p = run_path(name)
    if not p:
        return {}
    try:
        with open(os.path.join(p, "info.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}
