import os
import json
import pandas as pd
from tqdm import tqdm

from utils.llm_client import LLMClient
from utils.ecf_utils import load_ecf
from data_loader import DataLoader
from evaluator import Evaluator
from retrieval.allf_index import build_allf_index
from retrieval.allf_retrieval import run_allf_retrieval
import models
from query_expansion import (
    generate_hyde,
    generate_keywords,
    generate_emb_text,
    generate_reformulations,
    combine_queries
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'all_runs')
TEMP_RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'TempRunResults.tsv')
QRELS_PATH = os.path.join(PROJECT_ROOT, 'qrels', 'formal-folder-qrel.txt')
TOPICS_PATH = os.path.join(PROJECT_ROOT, 'src', 'data_creation', 'topics_output.txt')

def prepare_queries(llm_client):
    """Generates all expanded queries if they do not exist."""
    print("Preparing queries...")
    with open(TOPICS_PATH, 'r', encoding='utf-8') as f:
        topics_data = json.load(f)
    topics = [{'ID': tid, **info} for tid, info in topics_data.items()]
    
    hyde_path = os.path.join(DATA_DIR, 'queries_hyde.json')
    kw_path = os.path.join(DATA_DIR, 'queries_keywords.json')
    emb_path = os.path.join(DATA_DIR, 'queries_embtext.json')
    ref_path = os.path.join(DATA_DIR, 'queries_reformulations.json')
    
    if not os.path.exists(hyde_path): generate_hyde.generate_hyde_queries(topics, llm_client, hyde_path)
    if not os.path.exists(kw_path): generate_keywords.generate_keywords_queries(topics, llm_client, kw_path)
    if not os.path.exists(emb_path): generate_emb_text.generate_emb_text_queries(topics, llm_client, emb_path)
    if not os.path.exists(ref_path): generate_reformulations.generate_reformulations_queries(topics, llm_client, ref_path)
    
    # Combine queries (always regenerate to pick up any changes)
    combine_queries.combine_queries(hyde_path, kw_path, emb_path, ref_path, TOPICS_PATH, DATA_DIR)
        
    print("Queries prepared.")
    return topics

def load_queries():
    def _load(name):
        path = os.path.join(DATA_DIR, f'queries_{name}.json')
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
        
    return {
        'Q1': _load('hyde'),
        'Q2': _load('keywords'),
        'Q3': _load('embtext'),
        'Q4': _load('reformulations'),
        'QALL': _load('qall'),
        'Q0+Q1': _load('q0q1'),
        'Q0+Q2': _load('q0q2'),
        'Q0+Q3': _load('q0q3'),
        'Q0+Q4': _load('q0q4'),
        'Q0+QALL': _load('q0qall')
    }

def run_experiment(run_id, model_name, config_name, query_type_name, topics, queries_dict, folder_metadata, evaluator, trained_models=None):
    """Executes a single run and evaluates it."""
    run_folder = os.path.join(RESULTS_DIR, f"Phase1_{run_id}_{config_name}_{model_name}")
    os.makedirs(run_folder, exist_ok=True)
    
    if trained_models is None:
        # Need to train models
        base_models = ['b'] if model_name == 'b' else ['e'] if model_name == 'e' else ['c'] if model_name == 'c' else list(model_name)
        active_models = {}
        for m in base_models:
            full_m = 'bm25' if m == 'b' else 'embeddings' if m == 'e' else 'colbert'
            active_models[m] = build_allf_index(folder_metadata, config_name, full_m, models)
    else:
        active_models = trained_models
        
    # Prepare query strings
    query_strs = {}
    if query_type_name == 'Q0':
        for t in topics:
            query_strs[t['ID']] = f"{t['TITLE']} {t['DESCRIPTION']}"
    else:
        query_strs = queries_dict[query_type_name]
        
    # Run retrieval
    results = run_allf_retrieval(query_strs, active_models, topics)
    
    # Save and evaluate
    evaluator.save_run_file(results, TEMP_RESULTS_PATH, run_id)
    json_path = os.path.join(run_folder, 'AllDocuments_TopicsFolderMetrics.json')
    evaluator.evaluate(TEMP_RESULTS_PATH, json_path)
    evaluator.generate_aggregated_metrics(run_folder, 'all_documents')
    
    # Return mean nDCG@5
    with open(os.path.join(run_folder, 'all_documents_model_overall_stats.json'), 'r') as f:
        stats = json.load(f)
    return stats.get('model_global_ndcg', {}).get('mean', 0.0), active_models

def run_phase1():
    print("=== Starting Phase 1: Query Expansion ===")
    # Initialize DataLoader to get folder metadata
    loader = DataLoader(PROJECT_ROOT)
    folder_metadata = loader.folder_metadata
    
    # Setup Evaluator
    evaluator = Evaluator(QRELS_PATH, QRELS_PATH) # using folder qrels for both to avoid errors if box qrel is needed
    
    llm = LLMClient(cache_dir=os.path.join(DATA_DIR, 'llm_cache'), provider='groq')
    topics = prepare_queries(llm)
    queries = load_queries()
    
    # SET 1A: Normal Queries x Folder Index Fields
    print("--- Running Set 1A ---")
    configs = ['F1', 'F2', 'F3', 'F4', 'F5']
    models_to_test = ['b', 'e', 'c', 'bce', 'bc', 'be', 'ce']
    
    results_1a = {} # {(config, model): ndcg}
    
    # We can cache the trained indices for each config so we don't rebuild them for every combination
    cached_indices = {} # {config: {'b': model_b, 'e': model_e, 'c': model_c}}
    
    run_counter = 1
    for config in configs:
        print(f"Building indices for {config}...")
        cached_indices[config] = {
            'b': build_allf_index(folder_metadata, config, 'bm25', models),
            'e': build_allf_index(folder_metadata, config, 'embeddings', models),
            'c': build_allf_index(folder_metadata, config, 'colbert', models)
        }
        
        for m in models_to_test:
            run_id = f"1A-{run_counter:02d}"
            print(f"Running {run_id}: {config} + {m}")
            # select the base models needed
            needed_models = {k: cached_indices[config][k] for k in list(m)}
            ndcg, _ = run_experiment(run_id, m, config, 'Q0', topics, queries, folder_metadata, evaluator, trained_models=needed_models)
            results_1a[(config, m)] = ndcg
            run_counter += 1
            
    # Determine F_best for each model
    f_best = {}
    for m in models_to_test:
        best_cfg = max(configs, key=lambda c: results_1a[(c, m)])
        f_best[m] = best_cfg
        print(f"F_best for {m}: {best_cfg} (nDCG@5: {results_1a[(best_cfg, m)]:.4f})")
        
    print("--- Running Set 1B ---")
    # SET 1B: Expanded Queries
    # Format: (run_id, query_type, model, config_type)
    # config_type: 'best' uses F_best[model], otherwise uses the literal config name
    set_1b_runs = [
        # 1B-1: HyDE (Q1) alone — all models
        ('1B-01', 'Q1', 'b', 'best'), ('1B-02', 'Q1', 'e', 'best'), ('1B-03', 'Q1', 'c', 'best'),
        ('1B-04', 'Q1', 'bce', 'best'), ('1B-05', 'Q1', 'bc', 'best'), ('1B-06', 'Q1', 'be', 'best'), ('1B-07', 'Q1', 'ce', 'best'),
        # 1B-2: Keywords (Q2) alone — B, BC, BCE
        ('1B-08', 'Q2', 'b', 'best'), ('1B-09', 'Q2', 'bc', 'best'), ('1B-10', 'Q2', 'bce', 'best'),
        # 1B-3: Embedding Text (Q3) alone — E, CE, BE, BCE
        ('1B-11', 'Q3', 'e', 'best'), ('1B-12', 'Q3', 'ce', 'best'), ('1B-13', 'Q3', 'be', 'best'), ('1B-14', 'Q3', 'bce', 'best'),
        # 1B-4: Reformulations (Q4) alone — all models
        ('1B-15', 'Q4', 'b', 'best'), ('1B-16', 'Q4', 'e', 'best'), ('1B-17', 'Q4', 'c', 'best'),
        ('1B-18', 'Q4', 'bce', 'best'), ('1B-19', 'Q4', 'bc', 'best'), ('1B-20', 'Q4', 'be', 'best'), ('1B-21', 'Q4', 'ce', 'best'),
        # 1B-5: Q0 + Individual LLM Expansion
        ('1B-22', 'Q0+Q1', 'b', 'best'), ('1B-23', 'Q0+Q1', 'e', 'best'), ('1B-24', 'Q0+Q1', 'bce', 'best'),
        ('1B-25', 'Q0+Q2', 'b', 'best'), ('1B-26', 'Q0+Q2', 'bce', 'best'),
        ('1B-27', 'Q0+Q3', 'e', 'best'), ('1B-28', 'Q0+Q3', 'bce', 'best'),
        ('1B-29', 'Q0+Q4', 'b', 'best'), ('1B-30', 'Q0+Q4', 'e', 'best'), ('1B-31', 'Q0+Q4', 'bce', 'best'),
        # 1B-6: Combined Expansions (LLM-only and Q0-anchored)
        ('1B-32', 'QALL', 'bce', 'best'),
        ('1B-33', 'Q0+QALL', 'bce', 'best'),
        ('1B-34', 'Q0+QALL', 'b', 'best'),
        ('1B-35', 'Q0+QALL', 'e', 'best'),
        # 1B-7: Query-Index Interactions (Q0+QALL × all configs)
        ('1B-36', 'Q0+QALL', 'bce', 'F1'),
        ('1B-37', 'Q0+QALL', 'bce', 'F2'),
        ('1B-38', 'Q0+QALL', 'bce', 'F3'),
        ('1B-39', 'Q0+QALL', 'bce', 'F4'),
        ('1B-40', 'Q0+QALL', 'bce', 'F5'),
    ]
    
    results_1b = {}
    for run_id, query_type, m, cfg_type in tqdm(set_1b_runs, desc="Set 1B Runs"):
        config = f_best[m] if cfg_type == 'best' else cfg_type
        print(f"Running {run_id}: {config} + {m} + {query_type}")
        needed_models = {k: cached_indices[config][k] for k in list(m)}
        ndcg, _ = run_experiment(run_id, m, config, query_type, topics, queries, folder_metadata, evaluator, trained_models=needed_models)
        results_1b[run_id] = ndcg
        
    print("=== Phase 1 Complete ===")

if __name__ == "__main__":
    run_phase1()
