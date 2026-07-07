import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import json
import os
import re
import sys

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from utils.ecf_utils import load_ecf, get_training_docs_per_folder
from augmentation.cache_manager import AugmentationCache
from run_phase2_generation import save_generation_log_for_seed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CACHE_PATH = os.path.join(DATA_DIR, 'augmentation_cache.json')
ECF_DIR = os.path.join(PROJECT_ROOT, 'ecf', 'random_generated')

def robust_extract(llm_output):
    if not llm_output or not llm_output.strip():
        return "", ""
        
    desc_match = re.search(r'\*?\*?DESCRIPTION:?\*?\*?\s*(.*?)(?=\*?\*?KEYWORDS:?\*?\*?|$)', llm_output, re.IGNORECASE | re.DOTALL)
    desc = desc_match.group(1).strip() if desc_match else ""
    
    kw_match = re.search(r'\*?\*?KEYWORDS:?\*?\*?\s*(.*)', llm_output, re.IGNORECASE | re.DOTALL)
    kw = kw_match.group(1).strip() if kw_match else ""
    
    desc = " ".join(desc.split())
    kw = " ".join(kw.split())
    
    return desc, kw

def fix_cache():
    print("Fixing augmentation cache...")
    with open(CACHE_PATH, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
        
    fixed_count = 0
    for k, v in cache_data.items():
        if k.startswith('2A_'):
            desc = v.get('desc', '').strip()
            kw = v.get('kw', '').strip()
            if not desc or not kw:
                response = v.get('response', '')
                if response:
                    new_desc, new_kw = robust_extract(response)
                    # Use new ones if they are not empty, else keep whatever was there
                    if new_desc or new_kw:
                        v['desc'] = new_desc if new_desc else desc
                        v['kw'] = new_kw if new_kw else kw
                        fixed_count += 1
                        
    if fixed_count > 0:
        with open(CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Fixed {fixed_count} entries in the cache.")
    else:
        print("No entries needed fixing in the cache.")

def regenerate_logs():
    print("Regenerating generation logs...")
    loader = DataLoader(PROJECT_ROOT)
    folder_metadata = loader.folder_metadata
    doc_metadata = loader.items
    
    cache = AugmentationCache(CACHE_PATH)
    output_dir = os.path.join(DATA_DIR, 'phase2_generation_logs')
    seeds = [42, 100, 333, 777, 999]
    
    for seed in seeds:
        print(f"Processing ECF Seed {seed}...")
        ecf_path = os.path.join(ECF_DIR, f'ECF_RANDOM_{seed}.json')
        if not os.path.exists(ecf_path):
            print(f"Warning: ECF file not found: {ecf_path}")
            continue
            
        ecf_docs = load_ecf(ecf_path)
        all_folders_with_docs = get_training_docs_per_folder(ecf_docs, doc_metadata)
        
        save_generation_log_for_seed(seed, folder_metadata, doc_metadata, all_folders_with_docs, cache, output_dir)
        
if __name__ == '__main__':
    fix_cache()
    regenerate_logs()
