"""Shared paths, collection context, topic/folder loading and output helpers for the DSPy experiments."""
import json
import os
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOPICS_PATH = os.path.join(PROJECT_ROOT, "src", "data_creation", "topics_output.txt")
FOLDERS_PATH = os.path.join(PROJECT_ROOT, "data", "folders_metadata", "FoldersV1.3.json")
LUNA_ROOT = os.path.join(PROJECT_ROOT, "data", "llm_calls_luna")
MODEL_NAME = "gpt-6-luna"

from src.dspy_experiments.collection_context import COLLECTION_CONTEXT as CTX

QUERY_TYPE_FIELDS = {
    "T": ["TITLE"],
    "TD": ["TITLE", "DESCRIPTION"],
    "TDN": ["TITLE", "DESCRIPTION", "NARRATIVE"],
}


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def load_topics():
    with open(TOPICS_PATH, encoding="utf-8") as f:
        return list(json.load(f).values())


def build_query_text(topic, fields):
    """Same input text the Llama runs received: one topic field per line."""
    return "\n".join(topic[f] for f in fields if topic.get(f))


def load_folders():
    with open(FOLDERS_PATH, encoding="utf-8") as f:
        return json.load(f)


def normalize_field(value):
    """Missing, blank or pandas-'nan' values all become None."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text == "" or text.lower() == "nan" else text


def build_folder_fields(folder):
    scope = normalize_field(folder.get("raw_scope")) or normalize_field(folder.get("scope_truncated"))
    return {
        "folder_label": normalize_field(folder.get("label")),
        "snc": normalize_field(folder.get("snc")),
        "parent_expanded_snc": normalize_field(folder.get("label_parent_expanded")),
        "scope_note": scope,
        "start_date": normalize_field(folder.get("date")),
        "end_date": normalize_field(folder.get("endDate")),
    }


def folder_inputs(fields):
    """Signature inputs for DescribeFolder; None is rendered as the text 'None', as before."""
    show = lambda v: v if v is not None else "None"
    return {
        "collection_context": CTX,
        "folder_label": show(fields["folder_label"]),
        "snc": show(fields["snc"]),
        "meaning": show(fields["parent_expanded_snc"]),
        "scope_note": show(fields["scope_note"]),
        "date_range": f"{show(fields['start_date'])} to {show(fields['end_date'])}",
    }


def usage_of(pred):
    """(input_tokens, output_tokens) summed over every LM the prediction used."""
    try:
        usage = pred.get_lm_usage() or {}
    except Exception:
        return None, None
    tin = sum(u.get("prompt_tokens", 0) or 0 for u in usage.values())
    tout = sum(u.get("completion_tokens", 0) or 0 for u in usage.values())
    return tin, tout


def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    """Atomic write: a kill mid-save never corrupts the file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)