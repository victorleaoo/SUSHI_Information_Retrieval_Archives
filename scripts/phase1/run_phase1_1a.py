"""
Phase 1 — Set 1A: Normal Queries × Folder Index Configurations.

Purpose: Identify the best folder metadata configuration (F1–F4) for AllF retrieval
         using the basic Q0 (Title + Description) query.

Runs: 28 experiments — {F1, F2, F3, F4} × {B, E, C, BCE, BC, BE, CE}

Results saved to: all_runs/Phase1/1A/<run_id>_<config>_<model>/
F_best per model saved to: all_runs/Phase1/1A/f_best.json

Usage:
  cd <repo_root>
  python scripts/run_phase1_1a.py [--skip-existing]

Logs are written to: data/logs/phase1_runs/1a_run_log.jsonl
"""
import sys
import os
import json
import argparse
import shutil
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import models as models_module
from retrieval.allf_index import build_allf_training_data
from retrieval.allf_retrieval import run_allf_retrieval
from evaluator import Evaluator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'all_runs', 'Phase1', '1A')
QRELS_PATH   = os.path.join(PROJECT_ROOT, 'qrels', 'formal-folder-qrel.txt')
TOPICS_PATH  = os.path.join(PROJECT_ROOT, 'src', 'data_creation', 'topics_output.txt')
FOLDERS_PATH = os.path.join(DATA_DIR, 'folders_metadata', 'FoldersV1.3.json')
LOGS_PATH    = os.path.join(DATA_DIR, 'logs', 'phase1_runs', '1a_run_log.jsonl')

CONFIGS  = ['F1', 'F2', 'F3', 'F4']
MODELS   = ['b', 'e', 'c', 'bce', 'bc', 'be', 'ce']

# Run table: 1A-01..28 in order (config × model)
RUN_TABLE = [
    (f'1A-{i:02d}', CONFIGS[(i - 1) // len(MODELS)], MODELS[(i - 1) % len(MODELS)])
    for i in range(1, len(CONFIGS) * len(MODELS) + 1)
]


def _log(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


def _section(title: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"\n{'='*65}")
    print(f"[{ts}] {title}")
    print('='*65)


def load_data():
    _section("Loading folders and topics")
    with open(FOLDERS_PATH, 'r', encoding='utf-8') as f:
        folders = json.load(f)
    _log(f"  Loaded {len(folders)} folders from {FOLDERS_PATH}")

    with open(TOPICS_PATH, 'r', encoding='utf-8') as f:
        topics_raw = json.load(f)
    topics = [{'ID': tid, **info} for tid, info in topics_raw.items()]
    _log(f"  Loaded {len(topics)} topics")
    return folders, topics


def build_query_dict(topics: list) -> dict:
    """Returns Q0 = Title + Description for each topic."""
    return {t['ID']: f"{t['TITLE']} {t['DESCRIPTION']}" for t in topics}


def build_models_for_config(config: str, training_data: list, run_id: str) -> dict:
    """Trains B, E, C models for a given F-config. Returns {'b': ..., 'e': ..., 'c': ...}."""
    _log(f"  Building BM25 (B) for {config}...")
    bm25 = models_module.BM25Model(['folderlabel'])
    bm25.train(training_data)

    _log(f"  Building Embeddings (E) for {config}...")
    emb = models_module.EmbeddingsModel()
    emb.train(training_data)

    colbert_idx = os.path.join(PROJECT_ROOT, 'data', 'indexes', f'colbert_1a_{run_id[:5]}_{config}')
    _log(f"  Building ColBERT (C) for {config} → index at {colbert_idx}...")
    col = models_module.ColBERTModel(index_path=colbert_idx)
    col.train(training_data)

    return {'b': bm25, 'e': emb, 'c': col}


def select_sub_models(model_key: str, all_models: dict) -> dict:
    """Returns the sub-dict needed for run_allf_retrieval."""
    mapping = {
        'b':   ['b'],
        'e':   ['e'],
        'c':   ['c'],
        'bc':  ['b', 'c'],
        'be':  ['b', 'e'],
        'ce':  ['c', 'e'],
        'bce': ['b', 'c', 'e'],
    }
    return {k: all_models[k] for k in mapping[model_key]}


def run_and_evaluate(run_id: str, config: str, model_key: str,
                     query_dict: dict, trained_models: dict,
                     topics: list, evaluator: Evaluator,
                     skip_existing: bool) -> float:
    """Runs one experiment, evaluates it, and returns nDCG@5."""
    run_folder = os.path.join(RESULTS_DIR, f'{run_id}_{config}_{model_key.upper()}')
    stats_file = os.path.join(run_folder, 'all_documents_model_overall_stats.json')

    if skip_existing and os.path.exists(stats_file):
        _log(f"  [{run_id}] SKIP (already exists): {run_folder}")
        with open(stats_file, 'r') as f:
            return json.load(f)['model_global_ndcg']['mean']

    os.makedirs(run_folder, exist_ok=True)
    _log(f"  [{run_id}] Running {config} × {model_key.upper()} ...")

    t0 = datetime.now()
    sub_models = select_sub_models(model_key, trained_models)
    results = run_allf_retrieval(query_dict, sub_models, topics)

    run_file = os.path.join(run_folder, 'run.txt')
    evaluator.save_run_file(results, run_file, run_name=run_id)

    metrics_file = os.path.join(run_folder, 'AllDocuments_TopicsFolderMetrics.json')
    evaluator.evaluate(run_file, metrics_file)
    evaluator.generate_aggregated_metrics(run_folder, 'all_documents')

    with open(stats_file, 'r') as f:
        ndcg = json.load(f)['model_global_ndcg']['mean']

    elapsed = (datetime.now() - t0).total_seconds()
    _log(f"  [{run_id}] DONE  nDCG@5 = {ndcg:.4f}  ({elapsed:.1f}s)")
    return ndcg


def write_run_log(entry: dict):
    os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
    with open(LOGS_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Run Phase 1 Set 1A experiments.')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip runs where results already exist.')
    args = parser.parse_args()

    run_start = datetime.now()
    _section("Phase 1 — Set 1A: Normal Query × Folder Index Configs")
    _log(f"  28 runs: {{F1,F2,F3,F4}} × {{B,E,C,BCE,BC,BE,CE}}")
    _log(f"  Results dir: {RESULTS_DIR}")
    _log(f"  Skip existing: {args.skip_existing}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    folders, topics = load_data()
    query_dict = build_query_dict(topics)
    evaluator = Evaluator(QRELS_PATH, QRELS_PATH)

    # Results matrix: {(config, model_key): ndcg}
    results_matrix = {}

    # Process one config at a time to avoid rebuilding indices unnecessarily
    for config in CONFIGS:
        _section(f"Building indices for config {config}")
        training_data = build_allf_training_data(folders, config)
        _log(f"  Training data: {len(training_data)} folder entries")

        trained_models = build_models_for_config(config, training_data, run_id=config)

        # Run all model combos for this config
        for run_id, cfg, model_key in RUN_TABLE:
            if cfg != config:
                continue

            _section(f"Experiment {run_id} | Config={config} | Model={model_key.upper()}")
            t_run = datetime.now()
            ndcg = run_and_evaluate(
                run_id, config, model_key, query_dict, trained_models,
                topics, evaluator, args.skip_existing,
            )
            results_matrix[(config, model_key)] = ndcg

            write_run_log({
                'timestamp': datetime.now().isoformat(),
                'run_id': run_id,
                'phase': '1A',
                'config': config,
                'model': model_key,
                'query_type': 'Q0',
                'ndcg_at_5': ndcg,
                'duration_s': (datetime.now() - t_run).total_seconds(),
            })

        # Free ColBERT index (GPU memory)
        del trained_models
        import gc; gc.collect()

    # ── Determine F_best per model ──────────────────────────────────────────────
    _section("Determining F_best per model")
    f_best = {}
    for model_key in MODELS:
        best_config = max(CONFIGS, key=lambda c: results_matrix.get((c, model_key), 0.0))
        best_score  = results_matrix.get((best_config, model_key), 0.0)
        f_best[model_key] = best_config
        _log(f"  {model_key.upper():4s}  F_best = {best_config}  nDCG@5 = {best_score:.4f}")

    f_best_path = os.path.join(RESULTS_DIR, 'f_best.json')
    with open(f_best_path, 'w') as f:
        json.dump(f_best, f, indent=2)
    _log(f"  F_best saved to {f_best_path}")

    # ── Summary table ───────────────────────────────────────────────────────────
    _section("Set 1A Summary Table")
    header = f"{'Run ID':8s}  {'Config':5s}  {'Model':6s}  {'nDCG@5':>8s}"
    print(header)
    print('-' * len(header))
    for run_id, config, model_key in RUN_TABLE:
        score = results_matrix.get((config, model_key), float('nan'))
        print(f"{run_id:8s}  {config:5s}  {model_key.upper():6s}  {score:8.4f}")

    elapsed = (datetime.now() - run_start).total_seconds()
    _section(f"Phase 1 Set 1A Complete ✓  (total {elapsed:.0f}s)")


if __name__ == '__main__':
    main()
