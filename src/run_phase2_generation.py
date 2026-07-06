import os
import json
import argparse
from tqdm import tqdm

from augmentation.aug_homophily import augment_with_neighbors, find_neighbor_folders, select_neighbor_docs
from utils.ecf_utils import load_ecf
from data_loader import DataLoader
from utils.llm_client import LLMClient
from augmentation.cache_manager import AugmentationCache
from augmentation.aug_base import select_docs_for_folder, augment_with_docs, augment_no_docs
from augmentation.aug_tcde import generate_tcde_label

def save_generation_log_for_seed(seed, folder_metadata, doc_metadata, all_folders_with_docs, cache, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_data = {}
    
    for fid, fmeta in folder_metadata.items():
        doc_ids = all_folders_with_docs.get(fid, [])
        has_docs = len(doc_ids) > 0
        
        folder_log = {
            "folder_meta": fmeta,
            "has_docs": has_docs,
            "training_docs_in_ecf": doc_ids,
            "generations": {}
        }
        
        # Helper to attach doc metadata
        def enrich_docs(doc_list):
            if not doc_list: return []
            res = []
            for item in doc_list:
                if isinstance(item, tuple):  # For 2B neighbor docs: (n_id, did)
                    did = item[1]
                    res.append({"folder_id": item[0], "doc_id": did, "title": doc_metadata.get(did, {}).get("title", ""), "summary": doc_metadata.get(did, {}).get("summary", "")})
                else:
                    res.append({"doc_id": item, "title": doc_metadata.get(item, {}).get("title", ""), "summary": doc_metadata.get(item, {}).get("summary", "")})
            return res
        
        # 2A
        key_2a = '2A_DOC' if has_docs else '2A_NODOC'
        aug_2a = cache.get(key_2a, fid, doc_ids if has_docs else None)
        if aug_2a:
            folder_log["generations"]["2A"] = {
                "desc": aug_2a.get("desc", ""),
                "kw": aug_2a.get("kw", ""),
                "selected_docs": enrich_docs(aug_2a.get("selected_docs", []))
            }
            
        # 2B
        if not has_docs:
            aug_2b = cache.get('2B_HOMO', fid)
            if aug_2b:
                folder_log["generations"]["2B"] = {
                    "desc": aug_2b.get("desc", ""),
                    "kw": aug_2b.get("kw", ""),
                    "match_level": aug_2b.get("match_level", ""),
                    "selected_neighbor_docs": enrich_docs(aug_2b.get("selected_neighbor_docs", []))
                }
                
        # 2C
        key_2c = '2C_TCDE'
        aug_2c = cache.get(key_2c, fid, doc_ids if has_docs else None)
        if aug_2c:
            folder_log["generations"]["2C"] = {
                "desc": aug_2c.get("desc", ""),
                "kw": aug_2c.get("kw", ""),
                "selected_docs": enrich_docs(aug_2c.get("selected_docs", []))
            }
            
        log_data[fid] = folder_log
        
    out_path = os.path.join(output_dir, f"generation_log_Seed_{seed}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)
    print(f"Saved generation log to {out_path}")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
ECF_DIR = os.path.join(PROJECT_ROOT, 'ecf', 'random_generated')
CACHE_PATH = os.path.join(DATA_DIR, 'augmentation_cache.json')

def run_generation(strategies=None):
    if strategies is None:
        strategies = ['2a', '2b', 'tcde']
        
    print(f"Initializing DataLoader... (Strategies to run: {strategies})")
    loader = DataLoader(PROJECT_ROOT)
    folder_metadata = loader.folder_metadata
    doc_metadata = loader.items
    
    llm = LLMClient(cache_dir=os.path.join(DATA_DIR, 'llm_cache'), provider='groq')
    cache = AugmentationCache(CACHE_PATH)
    
    # 5 Uniform ECF seeds
    seeds = [42, 100, 333, 777, 999]
    
    for seed in seeds:
        print(f"\nProcessing ECF Seed {seed}...")
        ecf_path = os.path.join(ECF_DIR, f'ECF_RANDOM_{seed}.json')
        if not os.path.exists(ecf_path):
            print(f"Warning: ECF file not found: {ecf_path}")
            continue
            
        ecf_docs = load_ecf(ecf_path)
        
        from utils.ecf_utils import get_training_docs_per_folder
        
        all_folders_with_docs = get_training_docs_per_folder(ecf_docs, doc_metadata)
            
        for fid, fmeta in tqdm(folder_metadata.items(), desc=f"Augmenting Folders (Seed {seed})"):
            doc_ids = all_folders_with_docs.get(fid, [])
            has_docs = len(doc_ids) > 0
            
            # --- 2A: Base Augmentation ---
            if '2a' in strategies:
                if has_docs:
                    cached_2a = cache.get('2A_DOC', fid, doc_ids)
                    if not cached_2a:
                        print(f"[{fid}] Generating 2A_DOC augmentation...")
                        selected_docs = select_docs_for_folder(fmeta.get('title', ''), doc_ids, doc_metadata)
                        desc, kw, prompt, response = augment_with_docs(fmeta, selected_docs, doc_metadata, llm)
                        cache.set('2A_DOC', fid, doc_ids, {'desc': desc, 'kw': kw, 'selected_docs': selected_docs, 'prompt': prompt, 'response': response})
                else:
                    cached_2a = cache.get('2A_NODOC', fid)
                    if not cached_2a:
                        print(f"[{fid}] Generating 2A_NODOC augmentation...")
                        desc, kw, prompt, response = augment_no_docs(fmeta, llm)
                        cache.set('2A_NODOC', fid, None, {'desc': desc, 'kw': kw, 'selected_docs': [], 'prompt': prompt, 'response': response})
                    
            # --- 2B: Homophily Augmentation ---
            # (Only applies to empty folders)
            if '2b' in strategies:
                if not has_docs:
                    cached_2b = cache.get('2B_HOMO', fid)
                    if not cached_2b:
                        print(f"[{fid}] Generating 2B_HOMO augmentation...")
                        neighbors, match_level = find_neighbor_folders(fmeta, all_folders_with_docs, folder_metadata)
                        if len(neighbors) > 0:
                            selected_neighbor_docs = select_neighbor_docs(neighbors)
                            desc, kw, prompt, response = augment_with_neighbors(fmeta, selected_neighbor_docs, match_level, doc_metadata, folder_metadata, llm)
                            cache.set('2B_HOMO', fid, None, {'desc': desc, 'kw': kw, 'match_level': match_level, 'selected_neighbor_docs': selected_neighbor_docs, 'prompt': prompt, 'response': response})
                        else:
                            print(f"[{fid}] No neighbors found for 2B_HOMO. Falling back to 2A_NODOC.")
                            # Fallback to 2A NODOC if no neighbors
                            cached_2a_nodoc = cache.get('2A_NODOC', fid)
                            if cached_2a_nodoc:
                                cache.set('2B_HOMO', fid, None, cached_2a_nodoc)
                            
            # --- 2C: TCDE Augmentation ---
            if 'tcde' in strategies:
                if has_docs:
                    cached_2c = cache.get('2C_TCDE', fid, doc_ids)
                    if not cached_2c:
                        print(f"[{fid}] Generating 2C_TCDE (DOCS) augmentation...")
                        selected_docs = select_docs_for_folder(fmeta.get('title', ''), doc_ids, doc_metadata)
                        desc, kw, prompt, response = generate_tcde_label(fmeta, selected_docs, doc_metadata, llm)
                        cache.set('2C_TCDE', fid, doc_ids, {'desc': desc, 'kw': kw, 'selected_docs': selected_docs, 'prompt': prompt, 'response': response})
                else:
                    cached_2c = cache.get('2C_TCDE', fid)
                    if not cached_2c:
                        print(f"[{fid}] Generating 2C_TCDE (NODOC) augmentation...")
                        desc, kw, prompt, response = generate_tcde_label(fmeta, None, doc_metadata, llm)
                        cache.set('2C_TCDE', fid, None, {'desc': desc, 'kw': kw, 'selected_docs': [], 'prompt': prompt, 'response': response})
                        
        # Save analysis log
        output_dir = os.path.join(DATA_DIR, 'phase2_generation_logs')
        save_generation_log_for_seed(seed, folder_metadata, doc_metadata, all_folders_with_docs, cache, output_dir)

    print("Generation complete for all 5 Uniform ECFs.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Phase 2 LLM Label Generation")
    parser.add_argument('--strategies', nargs='+', choices=['2a', '2b', 'tcde', 'all'], default=['all'], help="Strategies to run. Default is all.")
    args = parser.parse_args()
    
    if 'all' in args.strategies:
        strategies_to_run = ['2a', '2b', 'tcde']
    else:
        strategies_to_run = args.strategies
        
    run_generation(strategies_to_run)
