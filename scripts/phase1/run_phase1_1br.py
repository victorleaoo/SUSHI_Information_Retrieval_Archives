"""
Phase 1 — Set 1B-R: Revised Concept-Focused Queries.

Purpose: Test whether concept-focused query expansions (Q5-Q2R) outperform
         entity/event-focused expansions (Q1-Q4) for matching against folder labels.

Runs: 19 experiments.

Prerequisite: run_phase1_1a.py (for F_best) AND run_phase1_generate_queries.py (for revised queries)

Results saved to: all_runs/Phase1/1BR/<run_id>_<query_type>_<model>/

Usage:
  cd <repo_root>
  python scripts/run_phase1_1br.py [--skip-existing]

Logs: data/logs/phase1_runs/1br_run_log.jsonl
"""
import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import models as models_module
from retrieval.allf_index import build_allf_training_data
from retrieval.allf_retrieval import run_allf_retrieval
from evaluator import Evaluator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'all_runs', 'Phase1', '1BR')
RESULTS_1A   = os.path.join(PROJECT_ROOT, 'all_runs', 'Phase1', '1A')
QRELS_PATH   = os.path.join(PROJECT_ROOT, 'qrels', 'formal-folder-qrel.txt')
TOPICS_PATH  = os.path.join(PROJECT_ROOT, 'src', 'data_creation', 'topics_output.txt')
FOLDERS_PATH = os.path.join(DATA_DIR, 'folders_metadata', 'FoldersV1.3.json')
LOGS_PATH    = os.path.join(DATA_DIR, 'logs', 'phase1_runs', '1br_run_log.jsonl')

# Set 1B-R run definitions: (run_id, query_type, model_key)
# All use f_best config from 1A (no config override)
SET_1BR = [
    # 1BR-1: Individual Revised Expansions
    ('1BR-01', 'Q5',       'b'),
    ('1BR-02', 'Q5',       'e'),
    ('1BR-03', 'Q5',       'bce'),
    ('1BR-04', 'Q6',       'b'),
    ('1BR-05', 'Q6',       'e'),
    ('1BR-06', 'Q6',       'bce'),
    ('1BR-07', 'Q7',       'e'),
    ('1BR-08', 'Q7',       'bce'),
    ('1BR-09', 'Q2R',      'b'),
    ('1BR-10', 'Q2R',      'bce'),
    # 1BR-2: Q0-Augmented Revised Expansions
    ('1BR-11', 'Q0+Q5',    'b'),
    ('1BR-12', 'Q0+Q5',    'e'),
    ('1BR-13', 'Q0+Q5',    'bce'),
    ('1BR-14', 'Q0+Q6',    'b'),
    ('1BR-15', 'Q0+Q6',    'bce'),
    # 1BR-3: Combined Revised Expansions
    ('1BR-16', 'Q0+Q5+Q6', 'bce'),
    ('1BR-17', 'Q0+QALL-R','bce'),
    ('1BR-18', 'Q0+QALL-R','b'),
    ('1BR-19', 'Q0+QALL-R','e'),
]

QUERY_FILE_MAP = {
    'Q5':        'queries_concept.json',
    'Q6':        'queries_hyde_r.json',
    'Q7':        'queries_thematic.json',
    'Q2R':       'queries_keywords_r.json',
    'Q0+Q5':     'queries_q0q5.json',
    'Q0+Q6':     'queries_q0q6.json',
    'Q0+Q5+Q6':  'queries_q0q5q6.json',
    'Q0+QALL-R': 'queries_q0qall_r.json',
}


def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _section(title: str):
    print(f"\n{'='*65}\n[{datetime.now().strftime('%H:%M:%S')}] {title}\n{'='*65}")


def load_f_best() -> dict:
    f_best_path = os.path.join(RESULTS_1A, 'f_best.json')
    if not os.path.exists(f_best_path):
        raise FileNotFoundError(
            f"F_best file not found: {f_best_path}\nPlease run run_phase1_1a.py first."
        )
    with open(f_best_path) as f:
        f_best = json.load(f)
    _log(f"  Loaded F_best: {f_best}")
    return f_best


def load_queries() -> dict:
    queries = {}
    for qtype, fname in QUERY_FILE_MAP.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            _log(f"  WARNING: Missing {fname} — skipping {qtype}")
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            queries[qtype] = json.load(f)
        _log(f"  Loaded {qtype}: {len(queries[qtype])} topics")
    return queries


def build_models(config: str, folders: dict, run_id: str) -> dict:
    training_data = build_allf_training_data(folders, config)
    _log(f"  [{run_id}] {len(training_data)} entries for {config}")

    _log(f"  [{run_id}] Training BM25 ...")
    bm25 = models_module.BM25Model(['folderlabel'])
    bm25.train(training_data)

    _log(f"  [{run_id}] Training Embeddings ...")
    emb = models_module.EmbeddingsModel()
    emb.train(training_data)

    colbert_idx = os.path.join(PROJECT_ROOT, 'data', 'indexes', f'colbert_1br_{run_id}_{config}')
    _log(f"  [{run_id}] Training ColBERT → {colbert_idx} ...")
    col = models_module.ColBERTModel(index_path=colbert_idx)
    col.train(training_data)

    return {'b': bm25, 'e': emb, 'c': col}


def select_sub_models(model_key: str, all_models: dict) -> dict:
    parts = {'b': ['b'], 'e': ['e'], 'c': ['c'],
             'bc': ['b', 'c'], 'be': ['b', 'e'], 'ce': ['c', 'e'], 'bce': ['b', 'c', 'e']}
    return {k: all_models[k] for k in parts[model_key]}


def run_and_evaluate(run_id, query_type, model_key, config,
                     query_dict, trained_models, topics, evaluator, skip_existing):
    label      = f'{run_id}_{query_type.replace("+","").replace("-","").replace("/","")}_{model_key.upper()}'
    run_folder = os.path.join(RESULTS_DIR, label)
    stats_file = os.path.join(run_folder, 'all_documents_model_overall_stats.json')

    if skip_existing and os.path.exists(stats_file):
        _log(f"  [{run_id}] SKIP (exists)")
        with open(stats_file) as f:
            return json.load(f)['model_global_ndcg']['mean']

    os.makedirs(run_folder, exist_ok=True)
    t0 = datetime.now()

    sub_models = select_sub_models(model_key, trained_models)
    results    = run_allf_retrieval(query_dict, sub_models, topics)

    run_file = os.path.join(run_folder, 'run.txt')
    evaluator.save_run_file(results, run_file, run_name=run_id)
    evaluator.evaluate(run_file, os.path.join(run_folder, 'AllDocuments_TopicsFolderMetrics.json'))
    evaluator.generate_aggregated_metrics(run_folder, 'all_documents')

    with open(stats_file) as f:
        ndcg = json.load(f)['model_global_ndcg']['mean']

    elapsed = (datetime.now() - t0).total_seconds()
    _log(f"  [{run_id}] DONE  query={query_type}  model={model_key.upper()}  "
         f"config={config}  nDCG@5={ndcg:.4f}  ({elapsed:.1f}s)")
    return ndcg


def write_run_log(entry: dict):
    os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
    with open(LOGS_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Run Phase 1 Set 1B-R experiments.')
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    run_start = datetime.now()
    _section("Phase 1 — Set 1B-R: Revised Concept-Focused Queries (19 runs)")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    f_best  = load_f_best()
    queries = load_queries()

    with open(TOPICS_PATH, 'r', encoding='utf-8') as f:
        topics = [{'ID': tid, **info} for tid, info in json.load(f).items()]
    with open(FOLDERS_PATH, 'r', encoding='utf-8') as f:
        folders = json.load(f)
    _log(f"  Loaded {len(folders)} folders, {len(topics)} topics")

    evaluator    = Evaluator(QRELS_PATH, QRELS_PATH)
    run_results  = {}

    # Group by config to avoid rebuilding models
    configs_needed = {}
    for run_id, query_type, model_key in SET_1BR:
        config = f_best.get(model_key, 'F2')
        configs_needed.setdefault(config, []).append((run_id, query_type, model_key))

    for config, runs_for_config in sorted(configs_needed.items()):
        _section(f"Building models for config {config} ({len(runs_for_config)} runs)")
        trained_models = build_models(config, folders, run_id=config)

        for run_id, query_type, model_key in runs_for_config:
            _section(f"Experiment {run_id} | {query_type} | {model_key.upper()} | {config}")

            if query_type not in queries:
                _log(f"  [{run_id}] SKIP — query file for '{query_type}' not found")
                continue

            t_run = datetime.now()
            ndcg = run_and_evaluate(
                run_id, query_type, model_key, config,
                queries[query_type], trained_models, topics, evaluator, args.skip_existing,
            )
            run_results[run_id] = ndcg
            write_run_log({
                'timestamp': datetime.now().isoformat(),
                'run_id': run_id,
                'phase': '1BR',
                'config': config,
                'model': model_key,
                'query_type': query_type,
                'ndcg_at_5': ndcg,
                'duration_s': (datetime.now() - t_run).total_seconds(),
            })

        del trained_models
        import gc; gc.collect()

    # ── Summary ─────────────────────────────────────────────────────────────────
    _section("Set 1B-R Summary")
    print(f"{'Run ID':9s}  {'Query':14s}  {'Model':6s}  {'Config':5s}  {'nDCG@5':>8s}")
    print('-' * 58)
    for run_id, query_type, model_key in SET_1BR:
        config = f_best.get(model_key, '?')
        score  = run_results.get(run_id, float('nan'))
        print(f"{run_id:9s}  {query_type:14s}  {model_key.upper():6s}  {config:5s}  {score:8.4f}")

    elapsed = (datetime.now() - run_start).total_seconds()
    _section(f"Phase 1 Set 1B-R Complete ✓  (total {elapsed:.0f}s)")


if __name__ == '__main__':
    main()
