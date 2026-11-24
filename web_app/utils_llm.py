"""
utils_llm.py
------------
Reads the LLM prompts and the LLM outputs for the 🤖 LLM Experiments page.

- Prompts are read from the code with `ast` (importing signatures.py would pull in dspy and load the folders
  file): src/dspy_experiments/signatures.py, collection_context.py, generation/web_notes.py, and the GEPA
  instructions in data/llm_calls_luna/programs/.
- The text that reached the rankers is rebuilt with the runner's own builders (src/augmentation_text.py).
"""

import ast
import collections
import inspect
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from augmentation_text import (build_augmentation_text, build_core_themes_augmentation_text,  # noqa: E402
                               build_folder_label_augmentation_text)

DSPY_DIR = os.path.join(SRC_DIR, "dspy_experiments")
SIGNATURES_PY = os.path.join(DSPY_DIR, "signatures.py")
COLLECTION_CONTEXT_PY = os.path.join(DSPY_DIR, "collection_context.py")
WEB_NOTES_PY = os.path.join(DSPY_DIR, "generation", "web_notes.py")
LUNA_ROOT = os.path.join(PROJECT_ROOT, "data", "llm_calls_luna")
PROGRAMS_DIR = os.path.join(LUNA_ROOT, "programs")
FOLDERS_PATH = os.path.join(PROJECT_ROOT, "data", "folders_metadata", "FoldersV1.3.json")
TOPICS_PATH = os.path.join(SRC_DIR, "data_creation", "topics_output.txt")

QT_FILE = {"T--": "T", "TD-": "TD", "TDN": "TDN"}
QUERY_TYPE_FIELDS = {"T": ["TITLE"], "TD": ["TITLE", "DESCRIPTION"], "TDN": ["TITLE", "DESCRIPTION", "NARRATIVE"]}
SEEDS = [1, 42, 100, 300, 333, 777, 999, 2025, 6159, 12345, 19865, 53819, 56782, 62537, 72738, 75259, 81236,
         91823, 98665, 98765, 99009, 999777333, 120302, 123865, 170302, 180803, 5122025, 12052024, 12052025, 99000011]

# column -> query variant of build_augmentation_text (running_experiments.QUERY_VARIANTS)
QUERY_VARIANTS = {"BASE": None, "DOC": "DOC", "FL": "FL", "DOCFL": "AUG", "CT": "CT"}


# ──────────────────────────────────────────────────────────────────────────────
# Prompt extraction
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def _module_ast(path: str, mtime: float) -> ast.Module:
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read())


def _tree(path: str) -> ast.Module:
    return _module_ast(path, os.path.getmtime(path))


def _field_desc(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "desc":
            return ast.literal_eval(kw.value)
    return ""


def _is_field_call(node, kind: str) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == kind)


def _raw_signature(class_name: str) -> Dict:
    for node in _tree(SIGNATURES_PY).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            sig = {"instructions": inspect.cleandoc(ast.get_docstring(node, clean=False) or ""),
                   "inputs": [], "outputs": []}
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if _is_field_call(item.value, "InputField"):
                        sig["inputs"].append((item.target.id, _field_desc(item.value)))
                    elif _is_field_call(item.value, "OutputField"):
                        sig["outputs"].append((item.target.id, _field_desc(item.value)))
            return sig
    raise KeyError(f"signature {class_name} not found in signatures.py")


def load_constant(module_path: str, name: str):
    """A module-level constant: a string literal, or the `desc` of a dspy.InputField/OutputField call."""
    for node in _tree(module_path).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            if isinstance(node.value, ast.Call):
                return _field_desc(node.value)
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {module_path}")


# composed signatures (signatures.py: X.append("web_notes", _NOTES).with_instructions(X.instructions + "\n\n" + Y))
_WEB_COMPOSED = {
    "CoreThemesWeb": ("CoreThemes", "WEB_NOTES_QUERY"),
    "DescribeFolderWeb": ("DescribeFolder", "WEB_NOTES_FOLDER"),
    "DescribeFolderWithEvidenceWeb": ("DescribeFolderWithEvidence", "WEB_NOTES_FOLDER"),
}


def load_signature(class_name: str, n_topics: Optional[int] = None) -> Dict:
    """{"instructions": str, "inputs": [(name, desc)], "outputs": [(name, desc)]} as the code builds it."""
    if class_name in _WEB_COMPOSED:
        base, extra = _WEB_COMPOSED[class_name]
        sig = _raw_signature(base)
        sig = {"instructions": sig["instructions"] + "\n\n" + load_constant(SIGNATURES_PY, extra),
               "inputs": sig["inputs"] + [("web_notes", load_constant(SIGNATURES_PY, "_NOTES"))],
               "outputs": list(sig["outputs"])}
        return sig
    sig = _raw_signature(class_name)
    if class_name == "Doc2QueryFolder" and n_topics:
        sig["outputs"] = [(n, f"exactly {n_topics} search topics, each a noun phrase of 3-10 words")
                          if n == "search_topics" else (n, d) for n, d in sig["outputs"]]
    return sig


def collection_context() -> str:
    return load_constant(COLLECTION_CONTEXT_PY, "COLLECTION_CONTEXT")


def web_prompt(name: str) -> str:
    return load_constant(WEB_NOTES_PY, name)


def load_program_instruction(arm: str, split: str) -> str:
    path = os.path.join(PROGRAMS_DIR, f"{arm}_{split}.txt")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


PROGRAM_SIGNATURE = {"docs": "HypotheticalDocuments", "filing": "PredictFiling", "core_themes": "CoreThemes",
                     "folder": "DescribeFolder"}


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading LLM text…", max_entries=24)
def load_luna_json(rel_path: str) -> Dict:
    """A JSON file under data/llm_calls_luna/, cached by path ({} when missing)."""
    path = os.path.join(LUNA_ROOT, rel_path)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def luna_exists(rel_path: str) -> bool:
    return os.path.isfile(os.path.join(LUNA_ROOT, rel_path))


@st.cache_data
def load_folders_meta() -> Dict:
    with open(FOLDERS_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_topics() -> Dict:
    """{topic id: {TITLE, DESCRIPTION, NARRATIVE}}"""
    with open(TOPICS_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def code_list() -> List[str]:
    """PredictFiling's code_list input: 'CODE | MEANING' for every SNC used (signatures.py, top)."""
    meanings = collections.defaultdict(collections.Counter)
    for f in load_folders_meta().values():
        if f["snc"] != "Unknown" and f.get("label_parent_expanded"):
            meanings[f["snc"]][f["label_parent_expanded"]] += 1
    return [f"{code} | {c.most_common(1)[0][0]}" for code, c in sorted(meanings.items())]


def plain_folder_label(folder_id: str) -> str:
    """The folder-label text the rankers index without augmentation (run_generator: label_parent_expanded +
    scope_truncated, falling back to the raw label)."""
    meta = load_folders_meta().get(folder_id, {})
    try:
        return meta["label_parent_expanded"] + " " + meta["scope_truncated"]
    except Exception:
        return meta.get("label", folder_id)


def indexed_folder_text(folder_id: str, entry: Optional[Dict]) -> str:
    """The folder-label text the rankers indexed: plain label + '. ' + description (when valid)."""
    label = plain_folder_label(folder_id)
    if entry and entry.get("valid"):          # the runner keeps only entries with valid set
        aug = build_folder_label_augmentation_text(entry)
        if aug:
            return f"{label}. {aug}"
    return label


def base_query(topic: Dict, qt: str) -> str:
    """running_experiments.query_text: title / title. desc / title. desc. narr."""
    title, desc, narr = (topic.get(f, "") for f in ("TITLE", "DESCRIPTION", "NARRATIVE"))
    if qt == "TDN":
        return f"{title}. {desc}. {narr}".strip()
    if qt == "TD":
        return f"{title}. {desc}".strip()
    return title.strip()


def ranker_query(topic: Dict, qt: str, column: str, hip_entry: Optional[Dict], ct_entry: Optional[Dict]) -> str:
    """The query string the rankers received for one topic/formulation/column (invalid entries dropped)."""
    query = base_query(topic, qt)
    variant = QUERY_VARIANTS.get(column)
    if variant == "CT":
        entry = ct_entry or {}
        if entry and not (entry.get("core_themes") or {}).get("valid", True):
            entry = {}
        aug = build_core_themes_augmentation_text(entry)
    elif variant:
        entry = dict(hip_entry or {})
        for part in ("documents", "folder_label"):     # the runner drops invalid generations
            if part in entry and not (entry[part] or {}).get("valid", True):
                entry.pop(part)
        aug = build_augmentation_text(entry, variant)
    else:
        aug = ""
    return f"{query}. {aug}".strip() if aug else query


def query_files(query_set: str, qt: str) -> Tuple[str, str]:
    """(1_doc_folder_hip file, 2_core_themes file) of a query set directory (hw / opt_cv / opt_all / web)."""
    return (f"{query_set}/query_expansion/1_doc_folder_hip/{qt}.json",
            f"{query_set}/query_expansion/2_core_themes_and_related_concepts/{qt}.json")


def folds() -> Dict:
    return load_luna_json("folds.json")


def fold_of_topic(topic_id: str) -> Optional[str]:
    """The fold program whose text a topic is searched with (folds.json splits.<A|B>.apply_to)."""
    for split, d in folds().get("splits", {}).items():
        if topic_id in d.get("apply_to", []):
            return split
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Rerank
# ──────────────────────────────────────────────────────────────────────────────
def read_ranking_tsv(path: str, topic_id: str, depth: int = 20) -> List[Tuple[int, str, float]]:
    """[(rank, folder id, score)] of one topic from a runs/Random<s>.tsv file (topic, folder, rank, score, run)."""
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[0] == topic_id:
                rows.append((int(parts[2]), parts[1], float(parts[3])))
    rows.sort()
    return rows[:depth]


def rerank_candidate_line(n: int, folder_id: str, docs: Optional[List[str]] = None, scope_words: int = 40,
                          docs_per_candidate: int = 3) -> str:
    """rerank.candidate(): '[n] LABEL | MEANING | SCOPE NOTE' (+ '| document: TITLE - SUMMARY' parts)."""
    meta = load_folders_meta().get(folder_id, {})

    def norm(v):
        if v is None:
            return None
        t = str(v).strip()
        return None if t == "" or t.lower() == "nan" else t

    scope = norm(meta.get("raw_scope")) or norm(meta.get("scope_truncated"))
    scope = " ".join((scope or "None").split()[:scope_words])
    text = f"[{n}] {norm(meta.get('label')) or folder_id} | {norm(meta.get('label_parent_expanded')) or 'None'} | {scope}"
    if docs is not None:
        shown = docs[:docs_per_candidate]
        text += "".join(f" | document: {d}" for d in shown) if shown else " | documents: none"
    return text


@st.cache_data(show_spinner="Reading the seed's sample…")
def sampled_doc_ids(seed: int) -> List[str]:
    """Doc ids of one seed's pinned uniform sample (common.uniform_ecf_path), in file order."""
    path = os.path.join(PROJECT_ROOT, "ecf", "random_generated", f"ECF_UNIFORM_Random_Seed_{seed}.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        ecf = json.load(f)
    return [doc[-10:-4] for doc in ecf["ExperimentSets"][0]["TrainingDocuments"]]
