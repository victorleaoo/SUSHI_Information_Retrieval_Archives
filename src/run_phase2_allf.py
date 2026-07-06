import os
import json
import pandas as pd
from tqdm import tqdm
import shutil

from utils.ecf_utils import load_ecf
from data_loader import DataLoader
from evaluator import Evaluator
from retrieval.allf_retrieval import run_allf_retrieval
from augmentation.cache_manager import AugmentationCache
import models

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'all_runs')
QRELS_PATH = os.path.join(PROJECT_ROOT, 'qrels', 'formal-folder-qrel.txt')
TOPICS_PATH = os.path.join(PROJECT_ROOT, 'src', 'data_creation', 'topics_output.txt')
CACHE_PATH = os.path.join(DATA_DIR, 'augmentation_cache.json')
ECF_DIR = os.path.join(DATA_DIR, 'ECFs', 'Random_Uniform')

def get_augmented_text(fid, folder_meta, doc_ids, aug_method, cache: AugmentationCache):
    has_docs = len(doc_ids) > 0
    label = folder_meta.get('title', '')
    snc = folder_meta.get('snc', 'Unknown')
    expanded_snc = folder_meta.get('expanded_snc', label) if snc == "Unknown" or not snc.strip() else folder_meta.get('expanded_snc', '')
    scope = folder_meta.get('scope_note', '')
    
    parts = [label, expanded_snc]
    if scope and str(scope) != 'nan' and scope != "No scope note available for this SNC code.":
        parts.append(scope)
        
    aug_data = None
    if aug_method == 'Aug-Base':
        key = '2A_DOC' if has_docs else '2A_NODOC'
        aug_data = cache.get(key, fid, doc_ids if has_docs else None)
    elif aug_method == 'Aug-Homo':
        key = '2A_DOC' if has_docs else '2B_HOMO'
        aug_data = cache.get(key, fid, doc_ids if has_docs else None)
    elif aug_method == 'Aug-TCDE':
        key = '2C_TCDE'
        aug_data = cache.get(key, fid, doc_ids if has_docs else None)
    elif aug_method == 'Hybrid (2A-doc + 2C-nodoc)':
        key = '2A_DOC' if has_docs else '2C_TCDE'
        aug_data = cache.get(key, fid, doc_ids if has_docs else None)
    elif aug_method == 'Hybrid (2A-doc + 2B-nodoc)':
        key = '2A_DOC' if has_docs else '2B_HOMO'
        aug_data = cache.get(key, fid, doc_ids if has_docs else None)
        
    if aug_data:
        if 'desc' in aug_data and aug_data['desc']: parts.append(aug_data['desc'])
        if 'kw' in aug_data and aug_data['kw']: parts.append(aug_data['kw'])
            
    return ' | '.join(parts)

def build_augmented_allf_index(folders: dict, folder_to_docs: dict, aug_method: str, cache: AugmentationCache, model_name: str):
    training_data = []
    for fid, folder in folders.items():
        text = get_augmented_text(fid, folder, folder_to_docs[fid], aug_method, cache)
        training_entry = {
            'docno': fid,       
            'folder': fid,      
            'box': folder.get('box', ''),
            'date': folder.get('date', ''),
            'folderlabel': text,  
            'text_blob': text,    
        }
        training_data.append(training_entry)
        
    if model_name == 'bm25':
        model = models.BM25Model(['folderlabel']) 
    elif model_name == 'embeddings':
        model = models.EmbeddingsModel()
    elif model_name == 'colbert':
        model = models.ColBERTModel()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
        
    model.train(training_data)
    return model

def load_queries():
    def _load(name):
        path = os.path.join(DATA_DIR, f'queries_{name}.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f: return json.load(f)
        return {}
        
    q_tcde = _load('tcde')
    # Phase 2 decisions might have set Q_best_BCE to Q0+Q4, etc. We load them all to be safe.
    queries = {
        'Q1': _load('hyde'),
        'Q2': _load('keywords'),
        'Q3': _load('embtext'),
        'Q4': _load('reformulations'),
        'QALL': _load('qall'),
        'Q0+Q1': _load('q0q1'),
        'Q0+Q2': _load('q0q2'),
        'Q0+Q3': _load('q0q3'),
        'Q0+Q4': _load('q0q4'),
        'Q0+QALL': _load('q0qall'),
        'Q-TCDE': q_tcde
    }
    return queries

def run_experiment_seed(run_id, model_name, aug_method, query_type_name, topics, queries_dict, folder_metadata, folder_to_docs, cache, seed, evaluator):
    """Executes a single ECF seed for a Phase 2 run."""
    run_folder = os.path.join(RESULTS_DIR, f"Phase2_{run_id}_{aug_method}_{model_name}")
    os.makedirs(run_folder, exist_ok=True)
    
    # Needs to be re-built per ECF seed because augmented labels depend on ECF docs
    base_models = ['b'] if model_name == 'b' else ['e'] if model_name == 'e' else ['c'] if model_name == 'c' else list(model_name)
    active_models = {}
    for m in base_models:
        full_m = 'bm25' if m == 'b' else 'embeddings' if m == 'e' else 'colbert'
        active_models[m] = build_augmented_allf_index(folder_metadata, folder_to_docs, aug_method, cache, full_m)
        
    query_strs = {}
    if query_type_name == 'Q0':
        for t in topics:
            query_strs[t['ID']] = f"{t['TITLE']} {t['DESCRIPTION']}"
    else:
        # Handle Phase 1 Decision defaults if needed
        # Assume Q_best_B, etc are resolved to actual keys in queries_dict before passing here
        if query_type_name in queries_dict:
            query_strs = queries_dict[query_type_name]
        else:
            raise ValueError(f"Unknown query_type_name {query_type_name}")
            
    results = run_allf_retrieval(query_strs, active_models, topics)
    
    temp_path = os.path.join(PROJECT_ROOT, 'results', f'TempPhase2_{seed}.tsv')
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    evaluator.save_run_file(results, temp_path, run_id)
    
    # We move the run file to the run_folder/all_documents directory for the generate_aggregated_metrics
    seed_dir = os.path.join(run_folder, 'all_documents')
    os.makedirs(seed_dir, exist_ok=True)
    shutil.move(temp_path, os.path.join(seed_dir, f"{seed}.tsv"))
    
def run_phase2():
    print("=== Starting Phase 2: Folder Label Augmentation ===")
    loader = DataLoader(PROJECT_ROOT)
    folder_metadata = loader.folder_metadata
    doc_metadata = loader.items
    from utils.ecf_utils import get_training_docs_per_folder
    evaluator = Evaluator(QRELS_PATH, QRELS_PATH)
    cache = AugmentationCache(CACHE_PATH)
    
    with open(TOPICS_PATH, 'r', encoding='utf-8') as f:
        topics_data = json.load(f)
    topics = [{'ID': tid, **info} for tid, info in topics_data.items()]
    
    queries = load_queries()
    
    # Decisions from Phase 1. 
    # For now, we will simulate Q_best_BCE to 'Q0+QALL' (user can change this).
    Q_best_B = 'Q0+Q1'
    Q_best_E = 'Q0+Q3'
    Q_best_C = 'Q0+Q4'
    Q_best_BCE = 'Q0+QALL'
    
    def resolve_query(q):
        if q == 'Q_best_B': return Q_best_B
        if q == 'Q_best_E': return Q_best_E
        if q == 'Q_best_C': return Q_best_C
        if q == 'Q_best_BCE': return Q_best_BCE
        return q

    # Define runs: (Run ID, Augmentation, Query, Model)
    # We only run Uni for Phase 2 based on recent changes
    runs = [
        # 2A
        ('2A-01', 'Aug-Base', 'Q0', 'b'),
        ('2A-02', 'Aug-Base', 'Q0', 'e'),
        ('2A-03', 'Aug-Base', 'Q0', 'c'),
        ('2A-04', 'Aug-Base', 'Q0', 'bce'),
        ('2A-05', 'Aug-Base', 'Q0', 'bc'),
        ('2A-06', 'Aug-Base', 'Q0', 'be'),
        ('2A-07', 'Aug-Base', 'Q0', 'ce'),
        ('2A-08', 'Aug-Base', 'Q_best_B', 'b'),
        ('2A-09', 'Aug-Base', 'Q_best_E', 'e'),
        ('2A-10', 'Aug-Base', 'Q_best_C', 'c'),
        ('2A-11', 'Aug-Base', 'Q_best_BCE', 'bce'),
        ('2A-12', 'Aug-Base', 'Q_best_BCE', 'bc'),
        ('2A-13', 'Aug-Base', 'Q_best_BCE', 'be'),
        ('2A-14', 'Aug-Base', 'Q_best_BCE', 'ce'),
        
        # 2B
        ('2B-01', 'Aug-Homo', 'Q0', 'bce'),
        ('2B-02', 'Aug-Homo', 'Q_best_BCE', 'bce'),
        ('2B-03', 'Aug-Homo', 'Q0', 'b'),
        ('2B-04', 'Aug-Homo', 'Q_best_B', 'b'),
        ('2B-05', 'Aug-Homo', 'Q0', 'e'),
        ('2B-06', 'Aug-Homo', 'Q_best_E', 'e'),
        ('2B-07', 'Aug-Homo', 'Q0', 'c'),
        ('2B-08', 'Aug-Homo', 'Q_best_C', 'c'),
        
        # 2C
        ('2C-01', 'Aug-TCDE', 'Q0', 'bce'),
        ('2C-02', 'Aug-TCDE', 'Q_best_BCE', 'bce'),
        ('2C-03', 'Aug-TCDE', 'Q-TCDE', 'bce'),
        ('2C-04', 'Aug-TCDE', 'Q0', 'b'),
        ('2C-05', 'Aug-TCDE', 'Q_best_B', 'b'),
        ('2C-06', 'Aug-TCDE', 'Q0', 'e'),
        ('2C-07', 'Aug-TCDE', 'Q_best_E', 'e'),
        ('2C-08', 'Aug-TCDE', 'Q0', 'c'),
        ('2C-09', 'Aug-TCDE', 'Q_best_C', 'c'),
        
        # 2X
        ('2X-01', 'Hybrid (2A-doc + 2C-nodoc)', 'Q_best_BCE', 'bce'),
        ('2X-02', 'Hybrid (2A-doc + 2B-nodoc)', 'Q_best_BCE', 'bce')
    ]
    
    seeds = [42, 101, 202, 303, 404]
    
    for run_id, aug_method, query, m in runs:
        print(f"\n--- Running {run_id}: {aug_method} + {m} + {query} ---")
        actual_query = resolve_query(query)
        
        for seed in tqdm(seeds, desc=f"Evaluating ECF Seeds for {run_id}"):
            ecf_path = os.path.join(ECF_DIR, f'ECF_RANDOM_{seed}.json')
            if not os.path.exists(ecf_path): continue
            ecf_docs = load_ecf(ecf_path)
            folder_to_docs = get_training_docs_per_folder(ecf_docs, doc_metadata)
            
            run_experiment_seed(run_id, m, aug_method, actual_query, topics, queries, folder_metadata, folder_to_docs, cache, seed, evaluator)
            
        # After all seeds, run aggregation
        run_folder = os.path.join(RESULTS_DIR, f"Phase2_{run_id}_{aug_method}_{m}")
        if os.path.exists(os.path.join(run_folder, 'all_documents')):
            evaluator.generate_aggregated_metrics(run_folder, 'all_documents')
            with open(os.path.join(run_folder, 'all_documents_model_overall_stats.json'), 'r') as f:
                stats = json.load(f)
                ndcg = stats.get('model_global_ndcg', {}).get('mean', 0.0)
                ci = stats.get('model_global_ndcg', {}).get('margin', 0.0)
                print(f"Result {run_id}: {ndcg:.4f} ± {ci:.4f}")

    print("=== Phase 2 Complete ===")

if __name__ == '__main__':
    run_phase2()
