"""
Phase 1 — Set 1B: Original Expanded Queries.

Purpose: Identify the best query strategy using original (entity/event-focused)
         LLM expansions with F_best index config from Set 1A.

Runs: 40 experiments across 7 query strategies and 7 model combos.

Prerequisite: Run run_phase1_1a.py first (reads all_runs/Phase1/1A/f_best.json)
              AND run_phase1_generate_queries.py (reads data/queries_*.json)

Results saved to: all_runs/Phase1/1B/<run_id>_<query_type>_<model>/

Usage:
  cd <repo_root>
  python scripts/run_phase1_1b.py [--skip-existing]

Logs: data/logs/phase1_runs/1b_run_log.jsonl
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
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'all_runs', 'Phase1', '1B')
RESULTS_1A   = os.path.join(PROJECT_ROOT, 'all_runs', 'Phase1', '1A')
QRELS_PATH   = os.path.join(PROJECT_ROOT, 'qrels', 'formal-folder-qrel.txt')
TOPICS_PATH  = os.path.join(PROJECT_ROOT, 'src', 'data_creation', 'topics_output.txt')
FOLDERS_PATH = os.path.join(DATA_DIR, 'folders_metadata', 'FoldersV1.3.json')
LOGS_PATH    = os.path.join(DATA_DIR, 'logs', 'phase1_runs', '1b_run_log.jsonl')

# Set 1B run definitions: (run_id, query_type, model_key, config_override)
# config_override = None → use f_best[model_key]; 'F1'..'F5' → force that config
SET_1B = [
    # 1B-1: HyDE (Q1) — all models
    ('1B-01', 'Q1',      'b',   None),
    ('1B-02', 'Q1',      'e',   None),
    ('1B-03', 'Q1',      'c',   None),
    ('1B-04', 'Q1',      'bce', None),
    ('1B-05', 'Q1',      'bc',  None),
    ('1B-06', 'Q1',      'be',  None),
    ('1B-07', 'Q1',      'ce',  None),
    # 1B-2: BM25 Keywords (Q2) — B, BC, BCE only
    ('1B-08', 'Q2',      'b',   None),
    ('1B-09', 'Q2',      'bc',  None),
    ('1B-10', 'Q2',      'bce', None),
    # 1B-3: Embedding Text (Q3) — E, CE, BE, BCE
    ('1B-11', 'Q3',      'e',   None),
    ('1B-12', 'Q3',      'ce',  None),
    ('1B-13', 'Q3',      'be',  None),
    ('1B-14', 'Q3',      'bce', None),
    # 1B-4: Reformulations (Q4) — all models
    ('1B-15', 'Q4',      'b',   None),
    ('1B-16', 'Q4',      'e',   None),
    ('1B-17', 'Q4',      'c',   None),
    ('1B-18', 'Q4',      'bce', None),
    ('1B-19', 'Q4',      'bc',  None),
    ('1B-20', 'Q4',      'be',  None),
    ('1B-21', 'Q4',      'ce',  None),
    # 1B-5: Real Query + Individual LLM Expansion (Q0+Qx)
    ('1B-22', 'Q0+Q1',   'b',   None),
    ('1B-23', 'Q0+Q1',   'e',   None),
    ('1B-24', 'Q0+Q1',   'bce', None),
    ('1B-25', 'Q0+Q2',   'b',   None),
    ('1B-26', 'Q0+Q2',   'bce', None),
    ('1B-27', 'Q0+Q3',   'e',   None),
    ('1B-28', 'Q0+Q3',   'bce', None),
    ('1B-29', 'Q0+Q4',   'b',   None),
    ('1B-30', 'Q0+Q4',   'e',   None),
    ('1B-31', 'Q0+Q4',   'bce', None),
    # 1B-6: Combined Expansions
    ('1B-32', 'QALL',    'bce', None),
    ('1B-33', 'Q0+QALL', 'bce', None),
    ('1B-34', 'Q0+QALL', 'b',   None),
    ('1B-35', 'Q0+QALL', 'e',   None),
    # 1B-7: Query-Index Interactions (Q0+QALL × all configs)
    ('1B-36', 'Q0+QALL', 'bce', 'F1'),
    ('1B-37', 'Q0+QALL', 'bce', 'F2'),
    ('1B-38', 'Q0+QALL', 'bce', 'F3'),
    ('1B-39', 'Q0+QALL', 'bce', 'F4'),
    ('1B-40', 'Q0+QALL', 'bce', 'F5'),
]

QUERY_FILE_MAP = {
    'Q1':      'queries_hyde.json',
    'Q2':      'queries_keywords.json',
    'Q3':      'queries_embtext.json',
    'Q4':      'queries_reformulations.json',
    'Q0+Q1':   'queries_q0q1.json',
    'Q0+Q2':   'queries_q0q2.json',
    'Q0+Q3':   'queries_q0q3.json',
    'Q0+Q4':   'queries_q0q4.json',
    'QALL':    'queries_qall.json',
    'Q0+QALL': 'queries_q0qall.json',
}


def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _section(title: str):
    print(f"\n{'='*65}\n[{datetime.now().strftime('%H:%M:%S')}] {title}\n{'='*65}")


def load_f_best() -> dict:
    f_best_path = os.path.join(RESULTS_1A, 'f_best.json')
    if not os.path.exists(f_best_path):
        raise FileNotFoundError(
            f"F_best file not found at {f_best_path}\n"
            "Please run run_phase1_1a.py first."
        )
    with open(f_best_path) as f:
        f_best = json.load(f)
    _log(f"  Loaded F_best: {f_best}")
    return f_best


def load_queries() -> dict:
    """Loads all query files. Returns {query_type: {topic_id: query_string}}."""
    queries = {}
    for qtype, fname in QUERY_FILE_MAP.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            _log(f"  WARNING: Missing query file {fname} — skipping {qtype}")
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            queries[qtype] = json.load(f)
        _log(f"  Loaded {qtype}: {len(queries[qtype])} topics")
    return queries


def build_index_and_models(config: str, folders: dict, run_id: str) -> dict:
    """Builds and returns trained B, E, C models for the given config."""
    training_data = build_allf_training_data(folders, config)
    _log(f"  [{run_id}] Training data: {len(training_data)} entries for {config}")

    _log(f"  [{run_id}] Training BM25 ...")
    bm25 = models_module.BM25Model(['folderlabel'])
    bm25.train(training_data)

    _log(f"  [{run_id}] Training Embeddings ...")
    emb = models_module.EmbeddingsModel()
    emb.train(training_data)

    colbert_idx = os.path.join(PROJECT_ROOT, 'data', 'indexes', f'colbert_1b_{run_id}_{config}')
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
    run_label  = f'{run_id}_{query_type.replace("+", "").replace("-", "")}_{model_key.upper()}'
    run_folder = os.path.join(RESULTS_DIR, run_label)
    stats_file = os.path.join(run_folder, 'all_documents_model_overall_stats.json')

    if skip_existing and os.path.exists(stats_file):
        _log(f"  [{run_id}] SKIP (exists)")
        with open(stats_file) as f:
            return json.load(f)['model_global_ndcg']['mean']

    os.makedirs(run_folder, exist_ok=True)
    t0 = datetime.now()

    sub_models = select_sub_models(model_key, trained_models)
    results = run_allf_retrieval(query_dict, sub_models, topics)

    run_file = os.path.join(run_folder, 'run.txt')
    evaluator.save_run_file(results, run_file, run_name=run_id)
    evaluator.evaluate(run_file, os.path.join(run_folder, 'AllDocuments_TopicsFolderMetrics.json'))
    evaluator.generate_aggregated_metrics(run_folder, 'all_documents')

    with open(stats_file) as f:
        ndcg = json.load(f)['model_global_ndcg']['mean']

    elapsed = (datetime.now() - t0).total_seconds()
    _log(f"  [{run_id}] DONE  config={config}  query={query_type}  model={model_key.upper()}"
         f"  nDCG@5={ndcg:.4f}  ({elapsed:.1f}s)")
    return ndcg


def write_run_log(entry: dict):
    os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
    with open(LOGS_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Run Phase 1 Set 1B experiments.')
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    run_start = datetime.now()
    _section("Phase 1 — Set 1B: Original Expanded Queries (40 runs)")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    f_best  = load_f_best()
    queries = load_queries()

    with open(TOPICS_PATH, 'r', encoding='utf-8') as f:
        topics_raw = json.load(f)
    topics = [{'ID': tid, **info} for tid, info in topics_raw.items()]

    with open(FOLDERS_PATH, 'r', encoding='utf-8') as f:
        folders = json.load(f)
    _log(f"  Loaded {len(folders)} folders, {len(topics)} topics")

    evaluator = Evaluator(QRELS_PATH, QRELS_PATH)
    run_results = {}

    # Group runs by (config) to avoid rebuilding indices unnecessarily
    # Determine which config each run uses
    def get_config(run_def, f_best):
        _, _, model_key, config_override = run_def
        if config_override is not None:
            return config_override
        return f_best.get(model_key, 'F2')

    # Collect unique configs needed
    configs_needed = {}  # config → list of run defs
    for run_def in SET_1B:
        c = get_config(run_def, f_best)
        configs_needed.setdefault(c, []).append(run_def)

    for config, runs_for_config in sorted(configs_needed.items()):
        _section(f"Building models for config {config} ({len(runs_for_config)} runs)")
        trained_models = build_index_and_models(config, folders, run_id=config)

        for run_def in runs_for_config:
            run_id, query_type, model_key, _ = run_def
            _section(f"Experiment {run_id} | {query_type} | {model_key.upper()} | {config}")

            if query_type not in queries:
                _log(f"  [{run_id}] SKIP — query file for {query_type} not found")
                continue

            query_dict = queries[query_type]
            t_run = datetime.now()
            ndcg = run_and_evaluate(
                run_id, query_type, model_key, config,
                query_dict, trained_models, topics, evaluator, args.skip_existing,
            )
            run_results[run_id] = ndcg
            write_run_log({
                'timestamp': datetime.now().isoformat(),
                'run_id': run_id,
                'phase': '1B',
                'config': config,
                'model': model_key,
                'query_type': query_type,
                'ndcg_at_5': ndcg,
                'duration_s': (datetime.now() - t_run).total_seconds(),
            })

        del trained_models
        import gc; gc.collect()

    # ── Summary ─────────────────────────────────────────────────────────────────
    _section("Set 1B Summary")
    print(f"{'Run ID':8s}  {'Query':12s}  {'Model':6s}  {'Config':5s}  {'nDCG@5':>8s}")
    print('-' * 55)
    for run_id, query_type, model_key, config_override in SET_1B:
        config = config_override if config_override else f_best.get(model_key, '?')
        score  = run_results.get(run_id, float('nan'))
        print(f"{run_id:8s}  {query_type:12s}  {model_key.upper():6s}  {config:5s}  {score:8.4f}")

    elapsed = (datetime.now() - run_start).total_seconds()
    _section(f"Phase 1 Set 1B Complete ✓  (total {elapsed:.0f}s)")


if __name__ == '__main__':
    main()
