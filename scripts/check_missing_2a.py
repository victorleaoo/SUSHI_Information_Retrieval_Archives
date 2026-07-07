import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import os
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOGS_DIR = os.path.join(PROJECT_ROOT, 'data', 'phase2_generation_logs')

def check_missing():
    seeds = [42, 100, 333, 777, 999]
    
    for seed in seeds:
        log_path = os.path.join(LOGS_DIR, f'generation_log_Seed_{seed}.json')
        if not os.path.exists(log_path):
            print(f"Seed {seed}: Log file not found.")
            continue
            
        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        missing_count = 0
        total_count = len(data)
        missing_fids = []
        
        for fid, folder_data in data.items():
            gen_2a = folder_data.get("generations", {}).get("2A", {})
            desc = gen_2a.get("desc", "").strip()
            kw = gen_2a.get("kw", "").strip()
            
            if not desc or not kw:
                missing_count += 1
                missing_fids.append(fid)
                
        print(f"Seed {seed}: {missing_count}/{total_count} folders missing 2A generation.")
        if missing_count > 0:
            print(f"  Missing FIDs (first 10): {missing_fids[:10]}")

if __name__ == '__main__':
    check_missing()
