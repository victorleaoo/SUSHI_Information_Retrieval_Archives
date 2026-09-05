"""
Generates hypothetical-document and hypothetical-folder-label query expansions
for every topic, for each of the T / TD / TDN query types.

Usage (from the project root, so it survives a terminal disconnect):
    nohup python -m src.llm_experiments.query_expansion.doc_folder_hip.generate \
        > data/llm_calls/query_expansion/generate.out 2>&1 &

Writes one file per query type to data/llm_calls/query_expansion/{T,TD,TDN}.json.
Each file is a dict keyed by topic ID; each topic holds the original topic
fields, the constructed query_text, and both a "documents" and a
"folder_label" result (raw_response text + parsed structured fields + token/
timing metadata). The script re-reads its own output file on startup and
skips topics already present, so it can be safely re-run/resumed; the
underlying LLMRunner cache also means a re-run never repeats an API call.
"""

import json
import os
from datetime import datetime

from src.llm_experiments.collection_context import COLLECTION_CONTEXT
from src.llm_experiments.llm_runner import LLMRunner
from src.llm_experiments.query_expansion.doc_folder_hip.parsing import (
    parse_documents_response,
    parse_folder_label_response,
)
from src.llm_experiments.query_expansion.doc_folder_hip.prompts import (
    render_documents_prompt,
    render_folder_label_prompt,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
TOPICS_PATH = os.path.join(PROJECT_ROOT, "src", "data_creation", "topics_output.txt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "llm_calls", "query_expansion")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "llm_calls", "cache")
LOG_PATH = os.path.join(PROJECT_ROOT, "data", "llm_calls", "calls.jsonl")

QUERY_TYPE_FIELDS = {
    "T": ["TITLE"],
    "TD": ["TITLE", "DESCRIPTION"],
    "TDN": ["TITLE", "DESCRIPTION", "NARRATIVE"],
}


def load_topics() -> list:
    with open(TOPICS_PATH, "r", encoding="utf-8") as f:
        return list(json.load(f).values())


def build_query_text(topic: dict, fields: list) -> str:
    return "\n".join(topic[field] for field in fields if topic.get(field))


def load_or_init_output(output_path: str, query_type: str, model: str) -> dict:
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "query_type": query_type,
        "model": model,
        "generated_at": datetime.now().isoformat(),
        "topics": {},
    }


def save_output(output_path: str, data: dict):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_call_result(entry: dict, parsed: dict) -> dict:
    return {
        "raw_response": entry.get("response", ""),
        "parsed": parsed,
        "input_tokens": entry.get("input_tokens"),
        "output_tokens": entry.get("output_tokens"),
        "duration_seconds": entry.get("duration_seconds"),
        "created_at": entry.get("created_at"),
    }


def generate_for_query_type(query_type: str, fields: list, runner: LLMRunner):
    output_path = os.path.join(OUTPUT_DIR, f"{query_type}.json")
    data = load_or_init_output(output_path, query_type, runner.model)
    topics = load_topics()

    for topic in topics:
        topic_id = topic["ID"]
        if topic_id in data["topics"]:
            continue

        query_text = build_query_text(topic, fields)

        documents_prompt = render_documents_prompt(COLLECTION_CONTEXT, query_text)
        documents_entry = runner.run_with_meta(documents_prompt)
        documents_result = build_call_result(documents_entry, parse_documents_response(documents_entry.get("response", "")))

        folder_label_prompt = render_folder_label_prompt(COLLECTION_CONTEXT, query_text)
        folder_label_entry = runner.run_with_meta(folder_label_prompt)
        folder_label_result = build_call_result(folder_label_entry, parse_folder_label_response(folder_label_entry.get("response", "")))

        data["topics"][topic_id] = {
            "topic_id": topic_id,
            "original_query": {field: topic.get(field, "") for field in fields},
            "query_text": query_text,
            "documents": documents_result,
            "folder_label": folder_label_result,
        }

        save_output(output_path, data)
        print(f"[{query_type}] {topic_id} done")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runner = LLMRunner(cache_dir=CACHE_DIR, log_path=LOG_PATH)

    for query_type, fields in QUERY_TYPE_FIELDS.items():
        generate_for_query_type(query_type, fields, runner)


if __name__ == "__main__":
    main()
