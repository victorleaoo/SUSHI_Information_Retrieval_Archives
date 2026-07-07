import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import os
import json

def format_run_name(name):
    # Mapping abbreviations to readable names
    mapping = {
        "ALLFL": "AllFolders",
        "CMN": "No LLM",
        "LLM": "LLM",
        "LP": "LabelParent",
        "LAB": "LabelParentExt",
        "RS": "RawScope",
        "ST": "ScopeTruncated",
        "EFT": "Title",
        "DES": "DenseSummary",
        "BKW": "BM25Keywords",
        "NEX": "NoExp",
        "EXP": "Expanded",
        "CKS": "Keywords",
        "DHS": "DenseSummary",
        "TD": "Title+Desc",
        "TDN": "Title+Desc+Narr",
        "T": "Title"
    }
    
    parts = name.split('_')
    readable_parts = []
    for part in parts:
        subparts = part.split('-')
        readable_subparts = [mapping.get(sp, sp) for sp in subparts]
        readable_parts.append('-'.join(readable_subparts))
        
    return ' | '.join(readable_parts)

def analyze_runs():
    all_runs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'all_runs'))
    
    if not os.path.exists(all_runs_dir):
        print(f"Directory not found: {all_runs_dir}")
        return

    results = []

    for run_name in os.listdir(all_runs_dir):
        run_dir = os.path.join(all_runs_dir, run_name)
        
        # Skip if it's not a directory
        if not os.path.isdir(run_dir):
            continue
            
        stats_file = os.path.join(run_dir, "all_documents_model_overall_stats.json")
        
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                mean = data.get("model_global_ndcg", {}).get("mean", 0.0)
                margin = data.get("model_global_ndcg", {}).get("margin", 0.0)
                
                results.append((mean, margin, run_name))
            except Exception as e:
                print(f"Error reading {stats_file}: {e}")

    # Sort descending by mean
    results.sort(key=lambda x: x[0], reverse=True)

    print(f"{'Mean':<10} | {'Margin':<10} | {'Run Name'}")
    print("-" * 120)
    for mean, margin, run_name in results:
        readable_name = format_run_name(run_name)
        print(f"{mean:.4f}     ± {margin:.4f}     | {readable_name}")

if __name__ == "__main__":
    analyze_runs()
