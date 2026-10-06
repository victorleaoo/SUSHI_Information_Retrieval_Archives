"""Step 2 (and Step 6): query-side generation for HyDE-docs, Filing and Core Themes.

Writes, per query type, the JSON shapes run_generator.py reads:
  <out>/query_expansion/1_doc_folder_hip/{T,TD,TDN}.json
  <out>/query_expansion/2_core_themes_and_related_concepts/{T,TD,TDN}.json

Usage (project root):
  nohup python -m src.dspy_experiments.generate_query_side --out hw \
      > data/llm_calls_luna/logs/query_hw.out 2>&1 &
"""
import argparse
import json
import os
from datetime import datetime

import dspy

from src.dspy_experiments import lm as _lm  # noqa: F401  (configures DSPy)
from src.dspy_experiments.common import (CTX, LUNA_ROOT, MODEL_NAME, QUERY_TYPE_FIELDS,
                                         build_query_text, load_json, load_topics, log,
                                         save_json, usage_of)
from src.dspy_experiments.signatures import (CODE_LIST, MEANING, CoreThemes, HypotheticalDocuments,
                                             PredictFiling, ok_core_themes, ok_docs, ok_filing,
                                             refined, run_batch)

ARMS = {
    # name: (signature, reward, extra inputs)
    "documents": (HypotheticalDocuments, ok_docs, {}),
    "folder_label": (PredictFiling, ok_filing, {"code_list": CODE_LIST}),
    "core_themes": (CoreThemes, ok_core_themes, {}),
}


def parsed_fields(arm, pred):
    if pred is None:
        return {}
    if arm == "documents":
        p = list(pred.passages or [])
        return {f"doc_{i}": (p[i - 1] if len(p) >= i else None) for i in (1, 2, 3)}
    if arm == "folder_label":
        codes = list(pred.snc_codes or [])
        return {"snc_codes": codes,
                "label_text": [MEANING[c] for c in codes if c in MEANING],   # catalogue wording
                "subject_terms": list(pred.subject_terms or [])}
    return {"core_themes": pred.core_themes, "related_concepts": list(pred.related_concepts or [])}


def record(arm, pred, valid, program):
    tin, tout = usage_of(pred) if pred is not None else (None, None)
    return {
        "raw_response": json.dumps(dict(pred), ensure_ascii=False) if pred is not None else "",
        "parsed": parsed_fields(arm, pred),
        "valid": valid,
        "program": program,
        "input_tokens": tin,
        "output_tokens": tout,
        "created_at": datetime.now().isoformat(),
    }


def generate(query_type, out_root, program, modules, threads, repair_rounds):
    fields = QUERY_TYPE_FIELDS[query_type]
    topics = load_topics()
    hip_path = os.path.join(out_root, "query_expansion", "1_doc_folder_hip", f"{query_type}.json")
    ct_path = os.path.join(out_root, "query_expansion", "2_core_themes_and_related_concepts", f"{query_type}.json")
    header = {"query_type": query_type, "model": MODEL_NAME, "generated_at": datetime.now().isoformat()}
    hip = load_json(hip_path, {**header, "topics": {}})
    ct = load_json(ct_path, {**header, "topics": {}})

    for t in topics:
        base = {"topic_id": t["ID"], "original_query": {f: t.get(f, "") for f in fields},
                "query_text": build_query_text(t, fields)}
        hip["topics"].setdefault(t["ID"], dict(base))
        ct["topics"].setdefault(t["ID"], dict(base))

    for arm, (signature, reward, extra) in ARMS.items():
        store = ct if arm == "core_themes" else hip
        module = modules[arm] if modules else refined(signature, reward)
        for round_ in range(repair_rounds + 1):
            todo = [t for t in topics if not store["topics"][t["ID"]].get(arm, {}).get("valid")]
            if not todo:
                break
            log(f"[{query_type}] {arm}: round {round_}, {len(todo)} topics")
            inputs = [{"collection_context": CTX,
                       "information_need": store["topics"][t["ID"]]["query_text"], **extra}
                      for t in todo]
            results = run_batch(module, reward, inputs, rollout_start=10 * round_, num_threads=threads)
            for t, (pred, valid) in zip(todo, results):
                store["topics"][t["ID"]][arm] = record(arm, pred, valid, program)
        n_valid = sum(store["topics"][t["ID"]].get(arm, {}).get("valid", False) for t in topics)
        log(f"[{query_type}] {arm}: {n_valid}/{len(topics)} valid")

    save_json(hip_path, hip)
    save_json(ct_path, ct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hw", help="subfolder of data/llm_calls_luna (hw, opt_all, ...)")
    ap.add_argument("--query-types", nargs="+", default=["T", "TD", "TDN"])
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--repair-rounds", type=int, default=1)
    args = ap.parse_args()
    out_root = os.path.join(LUNA_ROOT, args.out)
    with dspy.track_usage() as usage:
        for qt in args.query_types:
            generate(qt, out_root, program=args.out, modules=None,
                     threads=args.threads, repair_rounds=args.repair_rounds)
    save_json(os.path.join(out_root, "usage", "query_side.json"), usage.get_total_tokens())


if __name__ == "__main__":
    main()