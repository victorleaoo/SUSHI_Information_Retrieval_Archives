import os
import json
import pandas as pd
import re
import glob
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
import streamlit as st

# ==========================================
# CONSTANTES
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
EXPERIMENTS_ROOT_DIR = os.path.join(PROJECT_ROOT, "all_runs")
SUSHI_ROOT_DIR = EXPERIMENTS_ROOT_DIR
PATH_ECF = os.path.join(PROJECT_ROOT, 'ecf', 'random_generated', 'ECF_ALL_TRAINING_SET.json')

DIFFICULT_TOPICS = {24, 32, 39, 33, 27, 38, 42, 45, 41, 29, 40, 34, 9, 12, 6}
IMPOSSIBLE_TOPICS = {3, 8, 10, 13, 14, 17, 25, 26, 30, 31, 43}
ALL_KNOWN_TOPICS = {f"T{i}" for i in range(1, 46)}

COLOR_MAP = {
    'B': '#1f77b4',                          # Blue
    'C': '#ff7f0e',                          # Orange
    'E': '#2ca02c',                          # Green
    'BC': '#d62728',                         # Red
    'BE': '#9467bd',                         # Purple
    'CE': '#8c564b',                         # Brown
    'BCE': '#e377c2',                        # Pink
    'BM25': '#1f77b4',                       # Blue
    'COLBERT': '#ff7f0e',                    # Orange
    'EMBEDDINGS': '#2ca02c',                 # Green
    'BM25-COLBERT': '#d62728',               # Red
    'BM25-EMBEDDINGS': '#9467bd',            # Purple
    'BM25-EMBEDDINGS-COLBERT': '#8c564b',    # Brown
    'BM25-COLBERT-TUNED': '#e377c2',         # Pink
    'BM25-EMBEDDINGS-COLBERT-TUNED': '#17becf', # Cyan/Teal
    'BM25-EMBEDDINGS-COLBERT-TUNED-WRRF': '#bcbd22' # Olive/Yellow-Green
}

def parse_run_folder(folder_name: str) -> Optional[Dict[str, str]]:
    """
    Parses folder name assuming format: <search>_<expansion>_<query>_<model>
    Example: F_SB-SS_T_BM25-COLBERT
    """
    parts = folder_name.rsplit('_', 1)
    if len(parts) == 2:
        return {
            "config": parts[0],
            "model": parts[1],
            "full_name": folder_name
        }
    return {
        "config": folder_name,
        "model": "DEFAULT",
        "full_name": folder_name
    }

def natural_keys(text: str) -> List[Any]:
    """Split text into numeric and non-numeric parts for natural sorting."""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def normalize_topic_key(key: str) -> str:
    """Normalize topic keys to the 'T{n}' format when possible."""
    try:
        match = re.search(r'\d+$', key)
        if match:
            return f"T{int(match.group())}"
        return key
    except (AttributeError, ValueError):
        return key

def format_cell_content(metrics: Any) -> str:
    """Format metrics for HTML display in table cells."""
    if isinstance(metrics, (float, int)):
        return f"<b>{metrics:.3f}</b>"
    if isinstance(metrics, dict) and metrics.get('is_summary', False):
        mean_val = metrics.get('mean', 0)
        return f"<b>Mean: {mean_val:.3f}</b>"
    if isinstance(metrics, dict):
        ndcg = metrics.get('ndcg_cut_5', 0)
        r_top5 = metrics.get('count_relevant_in_top5_model', 0)
        r_train = metrics.get('count_relevant_folders_training', 0)
        r_total = metrics.get('count_relevant_folders_total', 0)
        h_top5 = metrics.get('count_highly_relevant_in_top5_model', 0)
        h_train = metrics.get('count_highly_relevant_folders_training', 0)
        h_total = metrics.get('count_highly_relevant_folders_total', 0)
        return (
            f"<b>{ndcg:.3f}</b><br>"
            f"<span style='font-size: 0.85em;'>R: {r_top5}/{r_train}/{r_total}</span><br>"
            f"<span style='font-size: 0.85em;'>H: {h_top5}/{h_train}/{h_total}</span>"
        )
    return str(metrics)

def load_json_safely(filepath: str) -> Dict:
    """Load JSON from filepath; return empty dict on error."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def get_topic_title_map() -> Dict[str, str]:
    """Loads ECF topics and returns mapping from 'T1'..'T45' to 'T1: Topic Title'."""
    ecf_data = load_json_safely(PATH_ECF)
    all_topics = {}
    if 'ExperimentSets' in ecf_data:
        for es in ecf_data['ExperimentSets']:
            if 'Topics' in es:
                all_topics.update(es['Topics'])

    title_map = {}
    for tid, data in all_topics.items():
        match = re.search(r'\d+$', tid)
        if match:
            num = int(match.group())
            title = data.get('TITLE', '')
            t_key = f"T{num}"
            title_map[t_key] = f"{t_key}: {title}" if title else t_key

    for i in range(1, 46):
        t_key = f"T{i}"
        if t_key not in title_map:
            title_map[t_key] = t_key

    return title_map

def is_subfolder_match(target_subfolder: Optional[str], current_subfolder: str) -> bool:
    """
    Checks if current_subfolder matches target_subfolder (exact match or parent phase prefix).
    """
    if not target_subfolder or target_subfolder == "All Subfolders":
        return True
    if current_subfolder == target_subfolder:
        return True
    if current_subfolder.startswith(target_subfolder + "/"):
        return True
    return False

def get_run_folder_info_map() -> Dict[str, Dict[str, str]]:
    """
    Recursively scans EXPERIMENTS_ROOT_DIR and returns a dictionary mapping
    keys -> {'path': full_absolute_path, 'subfolder': relative_subfolder, 'name': folder_name}
    for all run folders containing metric JSON files.
    """
    if not os.path.exists(EXPERIMENTS_ROOT_DIR):
        return {}
    
    mapping = {}
    valid_metric_files = [
        'topics_mean_margin.json',
        'model_overall_stats.json',
        'all_documents_model_overall_stats.json',
        'AllDocuments_TopicsFolderMetrics.json',
        'topics_values.json',
        'folder_overall_stats.json'
    ]
    for root, dirs, files in os.walk(EXPERIMENTS_ROOT_DIR):
        dirs.sort(key=natural_keys)
        if any(f in files for f in valid_metric_files):
            fname = os.path.basename(root)
            rel_p = os.path.relpath(root, EXPERIMENTS_ROOT_DIR)
            parts = rel_p.split(os.sep)
            subfolder = '/'.join(parts[:-1]) if len(parts) > 1 else 'Root'
            info = {'path': root, 'subfolder': subfolder, 'name': fname}
            mapping[rel_p] = info
            if fname not in mapping or '_seed' in mapping[fname]['subfolder']:
                mapping[fname] = info
            
    return mapping

def get_run_folder_map() -> Dict[str, str]:
    """Returns a dict mapping folder_name -> full_path."""
    info_map = get_run_folder_info_map()
    res = {}
    for k, info in info_map.items():
        res[k] = info['path']
        fname = info['name']
        if fname not in res or '_seed' in res[fname]:
            res[fname] = info['path']
    return res

def resolve_run_folder_path(run_name: str, subfolder: Optional[str] = None) -> Optional[str]:
    """
    Resolves the exact directory path for a run_name, optionally scoped by subfolder.
    """
    info_map = get_run_folder_info_map()
    
    # 1. If run_name is an exact key in info_map (e.g. relative path)
    if run_name in info_map:
        info = info_map[run_name]
        if not subfolder or subfolder == "All Subfolders" or is_subfolder_match(subfolder, info['subfolder']):
            return info['path']

    # 2. Match both info['name'] == run_name and subfolder scope (preferring non-seed directories)
    if subfolder and subfolder != "All Subfolders":
        best_path = None
        for info in info_map.values():
            if info['name'] == run_name and is_subfolder_match(subfolder, info['subfolder']):
                if best_path is None or '_seed' in best_path:
                    best_path = info['path']
        if best_path:
            return best_path

    # 3. Fallback to get_run_folder_map() short-name lookup
    return get_run_folder_map().get(run_name)

def get_available_subfolders() -> List[str]:
    """Returns a sorted list of unique subfolder paths and top-level parent phases present inside EXPERIMENTS_ROOT_DIR."""
    info_map = get_run_folder_info_map()
    raw_subfolders = {info['subfolder'] for info in info_map.values()}
    
    parent_phases = set()
    for sf in raw_subfolders:
        if '/' in sf:
            parent = sf.split('/')[0]
            parent_phases.add(parent)
            
    all_scopes = sorted(list(raw_subfolders | parent_phases), key=natural_keys)
    return all_scopes

def get_grouped_run_configurations(subfolder: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Scans all run folders in EXPERIMENTS_ROOT_DIR and groups them by configuration name.
    If subfolder is provided and not 'All Subfolders', only runs matching that scope are included.
    Returns a dict mapping config_name -> list of run_folder_names.
    """
    info_map = get_run_folder_info_map()
    grouped: Dict[str, List[str]] = {}
    seen_paths = set()
    
    for key, info in sorted(info_map.items()):
        folder_path = info['path']
        if folder_path in seen_paths:
            continue
        if not is_subfolder_match(subfolder, info['subfolder']):
            continue
        
        seen_paths.add(folder_path)
        fname = info['name']
        if '_' in fname:
            cfg, _ = fname.rsplit('_', 1)
        else:
            cfg = fname
        grouped.setdefault(cfg, []).append(fname)
    return grouped

def process_experiment_data(selected_configs: List[str], grouped_runs: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, Dict]]:
    """
    Processes metric data for selected configurations.
    Returns:
      - df_chart: DataFrame with columns ['Topic', 'Topic Label', 'Type', 'nDCG', 'min_ci', 'max_ci']
      - df_table_dummy: Empty DataFrame (placeholder)
      - all_topics: List of topic keys sorted naturally (e.g. ['T1', 'T2', ...])
      - model_results: Dict mapping model_key -> {'stats': {'val': float, 'margin': float}, 'count': int}
    """
    folder_map = get_run_folder_map()
    title_map = get_topic_title_map()
    chart_rows = []
    model_results = {}
    all_topics_set = set()

    for config in selected_configs:
        run_names = grouped_runs.get(config, [])
        for fname in run_names:
            fpath = folder_map.get(fname)
            if not fpath or not os.path.exists(fpath):
                continue
            
            parts = fname.rsplit('_', 1)
            model_key = parts[1] if len(parts) == 2 else fname

            # Overall stats
            g = load_overall_stats(fpath)
            mean_val = g.get('mean', 0.0)
            margin_val = g.get('margin', 0.0)

            # Count seeds (Random*.json files)
            random_count = len([f for f in os.listdir(fpath) if f.startswith('Random') and f.endswith('.json')])
            if random_count == 0:
                random_count = 1

            model_results[model_key] = {
                'stats': {'val': mean_val, 'margin': margin_val},
                'count': random_count
            }

            # Per-topic margin data
            topic_data = load_margin_data(fpath)
            for t, vals in topic_data.items():
                if isinstance(vals, list) and len(vals) >= 3:
                    all_topics_set.add(t)
                    topic_label = title_map.get(t, t)
                    chart_rows.append({
                        'Topic': t,
                        'Topic Label': topic_label,
                        'Type': model_key,
                        'nDCG': vals[1],
                        'min_ci': vals[0],
                        'max_ci': vals[2]
                    })

    all_topics = sorted(list(all_topics_set), key=natural_keys)
    df_chart = pd.DataFrame(chart_rows)
    return df_chart, pd.DataFrame(), all_topics, model_results

def get_single_run_topic_chart_dataset(run_name: str, subfolder: Optional[str] = None) -> pd.DataFrame:
    """
    Extracts per-topic metric data for a single run folder.
    Returns DataFrame with columns ['Topic', 'Topic Label', 'nDCG', 'min_ci', 'max_ci', 'Relevance'].
    """
    title_map = get_topic_title_map()
    fpath = resolve_run_folder_path(run_name, subfolder)
    if not fpath or not os.path.exists(fpath):
        return pd.DataFrame()

    margin_p = os.path.join(fpath, 'topics_mean_margin.json')
    rel_p = os.path.join(fpath, 'topics_relevant_count_stats.json')

    topic_ndcg = load_json_safely(margin_p)
    topic_rel = load_json_safely(rel_p)

    all_topics = sorted(list(set(topic_ndcg.keys()) | set(topic_rel.keys())), key=natural_keys)

    rows = []
    for t in all_topics:
        vals = topic_ndcg.get(t, [0.0, 0.0, 0.0])
        rel_data = topic_rel.get(t, {})
        rel_mean = rel_data.get('mean', 0.0) if isinstance(rel_data, dict) else 0.0
        
        m_min = vals[0] if len(vals) >= 3 else 0.0
        val = vals[1] if len(vals) >= 3 else 0.0
        m_max = vals[2] if len(vals) >= 3 else 0.0

        rows.append({
            'Topic': t,
            'Topic Label': title_map.get(t, t),
            'nDCG': val,
            'min_ci': m_min,
            'max_ci': m_max,
            'Relevance': rel_mean
        })

    return pd.DataFrame(rows)

def calculate_folder_average(folder_path: str) -> Tuple[Optional[Dict[str, float]], List[str]]:
    """Compute per-topic mean nDCG from all Random*.json files in a folder."""
    if not os.path.exists(folder_path): return None, []
    valid_files = [f for f in os.listdir(folder_path) if f.startswith("Random") and f.endswith(".json")]
    if not valid_files: return None, []
    
    topic_accumulator = {topic: [] for topic in ALL_KNOWN_TOPICS}
    for filename in valid_files:
        data = load_json_safely(os.path.join(folder_path, filename))
        if not data: continue
        
        run_values = {}
        for raw_key, metrics in data.items():
            norm_key = normalize_topic_key(raw_key)
            val = metrics.get('ndcg_cut_5', 0.0)
            run_values[norm_key] = val
        
        for topic in ALL_KNOWN_TOPICS:
            topic_accumulator[topic].append(run_values.get(topic, 0.0))
            
    averages = {k: float(np.mean(v)) for k, v in topic_accumulator.items() if v}
    return averages, valid_files

def load_margin_data(folder_path: str) -> Dict:
    """Load per-topic mean/margin data from topics_mean_margin.json."""
    return load_json_safely(os.path.join(folder_path, "topics_mean_margin.json"))

def load_overall_stats(folder_path: str, filename: str = "model_overall_stats.json") -> Dict:
    """Load overall model stats JSON and return the model_global_ndcg section."""
    if not folder_path or not os.path.exists(folder_path):
        return {}
    for fn in [filename, "model_overall_stats.json", "all_documents_model_overall_stats.json", "folder_overall_stats.json"]:
        p = os.path.join(folder_path, fn)
        if os.path.exists(p):
            data = load_json_safely(p)
            if 'model_global_ndcg' in data:
                return data.get('model_global_ndcg', {})
            elif 'mean' in data:
                return data
    return {}

def get_model_metric_summary(stats_dict: Dict, topic_avgs_dict: Dict) -> Dict[str, float]:
    """Summarize model metrics using provided stats or topic averages."""
    if stats_dict and 'mean' in stats_dict:
        return {'val': stats_dict['mean'], 'margin': stats_dict.get('margin', 0.0)}
    elif topic_avgs_dict:
        return {'val': float(np.mean(list(topic_avgs_dict.values()))), 'margin': 0.0}
    else:
        return {'val': 0.0, 'margin': 0.0}

def build_multi_model_chart_dataset(topics: List[str], model_data: Dict[str, Dict]) -> pd.DataFrame:
    """
    Constructs the dataset for the Altair Chart.
    model_data structure: {'ModelName': {'avg': {...}, 'margins': {...}}}
    """
    chart_rows = []
    
    for t in topics:
        for model_name, data in model_data.items():
            avg_dict = data.get('avg', {})
            margin_dict = data.get('margins', {})
            
            if t in avg_dict:
                val = avg_dict[t]
                m_min, _, m_max = margin_dict.get(t, [val, val, val])
                
                chart_rows.append({
                    "Topic": t, 
                    "Type": model_name, 
                    "nDCG": val, 
                    "min_ci": m_min, 
                    "max_ci": m_max
                })
                
    return pd.DataFrame(chart_rows)

def build_chart_dataset(topics: List[str], avg_ex: Dict, margins_ex: Dict, avg_nex: Dict, margins_nex: Dict, oracle_data: Dict, sushi_data: Dict, avg_emb: Dict = {}, margins_emb: Dict = {}) -> pd.DataFrame:
    """Build per-topic chart rows across experiment variants."""
    chart_rows = []
    for t in topics:
        if avg_ex and t in avg_ex:
            val = avg_ex[t]
            m_min, _, m_max = margins_ex.get(t, [val, val, val])
            chart_rows.append({"Topic": t, "Type": "Avg With Expansion", "nDCG": val, "min_ci": m_min, "max_ci": m_max})
        if avg_nex and t in avg_nex:
            val = avg_nex[t]
            m_min, _, m_max = margins_nex.get(t, [val, val, val])
            chart_rows.append({"Topic": t, "Type": "Avg No Expansion", "nDCG": val, "min_ci": m_min, "max_ci": m_max})
        if avg_emb and t in avg_emb:
            val = avg_emb[t]
            m_min, _, m_max = margins_emb.get(t, [val, val, val])
            chart_rows.append({"Topic": t, "Type": "Embeddings (F_EMB_T)", "nDCG": val, "min_ci": m_min, "max_ci": m_max})
    return pd.DataFrame(chart_rows)

def get_configuration_ndcg_map(grouped_runs: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Computes the maximum nDCG mean score achieved across models for each configuration.
    """
    folder_map = get_run_folder_map()
    config_ndcg = {}
    for cfg, runs in grouped_runs.items():
        max_val = 0.0
        for r in runs:
            fpath = folder_map.get(r)
            if fpath:
                stats = load_overall_stats(fpath)
                val = stats.get("mean", 0.0)
                if val > max_val:
                    max_val = val
        config_ndcg[cfg] = max_val
    return config_ndcg

def get_all_runs_statistics(subfolder: Optional[str] = None) -> pd.DataFrame:
    """
    Scans EXPERIMENTS_ROOT_DIR recursively for all subfolders, looks for 
    overall stats, and counts Random_ files.
    """
    info_map = get_run_folder_info_map()
    rows = []
    seen_paths = set()
    
    for key, info in info_map.items():
        folder_path = info['path']
        if folder_path in seen_paths:
            continue
        if not is_subfolder_match(subfolder, info['subfolder']):
            continue

        seen_paths.add(folder_path)
        global_metrics = load_overall_stats(folder_path)
        
        mean_val = global_metrics.get("mean", 0.0)
        margin_val = global_metrics.get("margin", 0.0)
        
        random_count = len([
            f for f in os.listdir(folder_path) 
            if f.startswith("Random") and f.endswith(".json")
        ])
        
        rows.append({
            "Run Name": info['name'],
            "Subfolder": info['subfolder'],
            "Mean": mean_val,
            "Margin": margin_val,
            "Random Runs Count": random_count
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="Mean", ascending=False)
        
    return df

def load_relevance_stats(folder_path: str) -> Dict:
    """Loads the relevant count stats json."""
    return load_json_safely(os.path.join(folder_path, "topics_relevant_count_stats.json"))

def calculate_global_relevance_mean(topic_data: Dict[str, Dict]) -> float:
    """
    Calculates simple Global Mean from per-topic dictionaries.
    No margin needed for this counter metric.
    """
    if not topic_data:
        return 0.0
    
    means = [v.get('mean', 0.0) for v in topic_data.values()]
    if not means: return 0.0
    
    return float(np.mean(means))

def get_unified_comparison_dataframe(selected_run_names: List[str]) -> pd.DataFrame:
    """
    Builds the Master Table combining nDCG (with margin) and Relevance Counts (mean only).
    """
    rows = []
    
    all_topics = sorted(list(ALL_KNOWN_TOPICS), key=natural_keys)

    for run_name in selected_run_names:
        folder_path = resolve_run_folder_path(run_name)
        if not folder_path or not os.path.exists(folder_path): continue

        overall_stats = load_overall_stats(folder_path)
        topic_ndcg = load_margin_data(folder_path) 
        topic_rel = load_relevance_stats(folder_path)

        g_ndcg_mean = overall_stats.get('mean', 0.0)
        g_ndcg_margin = overall_stats.get('margin', 0.0)
        g_rel_mean = calculate_global_relevance_mean(topic_rel)

        row = {
            "Experiment Folder": run_name,
            "Global nDCG@5": f"{g_ndcg_mean:.4f} ± {g_ndcg_margin:.4f}",
            "Global Relevance": f"{g_rel_mean:.2f}"
        }

        for t in all_topics:
            if topic_ndcg and t in topic_ndcg:
                ndcg_val = topic_ndcg[t][1]
                ndcg_margin = topic_ndcg[t][2] - topic_ndcg[t][1]
            else:
                ndcg_val = 0.0
                ndcg_margin = 0.0
            
            if topic_rel and t in topic_rel:
                rel_val = topic_rel[t].get('mean', 0.0)
            else:
                rel_val = 0.0
            
            row[t] = f"{ndcg_val:.3f} ± {ndcg_margin:.3f} | {rel_val:.1f}"

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        cols = ["Experiment Folder", "Global nDCG@5", "Global Relevance"] + all_topics
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
    return df

def categorize_topics_comparison(df_run_a: pd.DataFrame, df_run_b: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Categorizes topics based on relative performance comparison between Run A and Run B:
    - 'better': Run A topic mean at least 10% better than Run B topic mean (pct_diff >= +0.10)
    - 'equal': Run A topic mean between 10% worse and 10% better (-0.10 < pct_diff < +0.10)
    - 'worse': Run A topic mean at least 10% worse than Run B topic mean (pct_diff <= -0.10)
    """
    if df_run_a.empty or df_run_b.empty:
        return {'better': pd.DataFrame(), 'equal': pd.DataFrame(), 'worse': pd.DataFrame()}

    df_a = df_run_a[['Topic', 'Topic Label', 'nDCG']].rename(columns={'nDCG': 'nDCG_A'})
    df_b = df_run_b[['Topic', 'nDCG']].rename(columns={'nDCG': 'nDCG_B'})

    merged = pd.merge(df_a, df_b, on='Topic', how='inner')
    merged['Diff'] = merged['nDCG_A'] - merged['nDCG_B']

    def calc_pct(row):
        a, b = row['nDCG_A'], row['nDCG_B']
        if b > 0:
            return (a - b) / b
        elif b == 0:
            return 1.0 if a > 0 else 0.0
        return 0.0

    merged['Pct_Diff'] = merged.apply(calc_pct, axis=1)

    better_df = merged[merged['Pct_Diff'] >= 0.10].sort_values(by='Pct_Diff', ascending=False)
    equal_df = merged[(merged['Pct_Diff'] > -0.10) & (merged['Pct_Diff'] < 0.10)].sort_values(by='Diff', ascending=False)
    worse_df = merged[merged['Pct_Diff'] <= -0.10].sort_values(by='Pct_Diff', ascending=True)

    return {
        'better': better_df,
        'equal': equal_df,
        'worse': worse_df
    }


def run_wilcoxon_test(run_name_a: str, run_name_b: str,
                      subfolder_a: Optional[str] = None,
                      subfolder_b: Optional[str] = None) -> Dict:
    """
    Runs a Wilcoxon signed-rank test comparing two runs across their seed files.
    For each common seed, computes the mean nDCG@5 across all topics, then
    performs a paired test on those seed-level means.

    Returns dict with: statistic, p_value, n_seeds, significant, winner,
    mean_a, mean_b, wins_a, wins_b.
    """
    from scipy import stats as sp_stats

    folder_a = resolve_run_folder_path(run_name_a, subfolder_a) or ""
    folder_b = resolve_run_folder_path(run_name_b, subfolder_b) or ""

    if not folder_a or not folder_b:
        return {"error": "Could not resolve run folder paths."}

    def _load_seed_means(folder_path: str) -> Dict[str, float]:
        """Load Random*.json files and return {seed_filename: mean_ndcg}."""
        seed_means = {}
        if not os.path.isdir(folder_path):
            return seed_means
        for fname in sorted(os.listdir(folder_path)):
            if fname.startswith("Random") and fname.endswith(".json"):
                fpath = os.path.join(folder_path, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                    scores = [
                        m.get('ndcg_cut_5', 0.0)
                        for m in content.values()
                        if isinstance(m, dict) and 'ndcg_cut_5' in m
                    ]
                    if scores:
                        seed_means[fname] = float(np.mean(scores))
                except Exception:
                    pass
        return seed_means

    seeds_a = _load_seed_means(folder_a)
    seeds_b = _load_seed_means(folder_b)

    common_seeds = sorted(set(seeds_a.keys()) & set(seeds_b.keys()))
    if len(common_seeds) < 2:
        return {"error": f"Not enough common seeds ({len(common_seeds)}) for Wilcoxon test. Need ≥ 2."}

    vals_a = [seeds_a[s] for s in common_seeds]
    vals_b = [seeds_b[s] for s in common_seeds]

    mean_a = float(np.mean(vals_a))
    mean_b = float(np.mean(vals_b))
    wins_a = sum(1 for a, b in zip(vals_a, vals_b) if a > b)
    wins_b = sum(1 for a, b in zip(vals_a, vals_b) if b > a)

    try:
        stat, p_val = sp_stats.wilcoxon(vals_a, vals_b)
        significant = p_val < 0.05
        winner = run_name_a if mean_a > mean_b else run_name_b if mean_b > mean_a else "Tie"
    except ValueError:
        stat, p_val, significant, winner = 0.0, 1.0, False, "Identical"

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "n_seeds": len(common_seeds),
        "significant": significant,
        "winner": winner if significant else "No significant difference",
        "mean_a": mean_a,
        "mean_b": mean_b,
        "wins_a": wins_a,
        "wins_b": wins_b,
    }


@st.cache_data(show_spinner="Computing seed variance…")
def compute_seed_variance_df(run_name: str, subfolder: Optional[str] = None) -> pd.DataFrame:
    """
    Computes per-topic nDCG@5 distributions across all available seeds for a single run.
    Returns a DataFrame with columns:
        Topic, Topic Title, Mean, Std, Min, Max, Range, N_Seeds, Values
    """
    folder_path = resolve_run_folder_path(run_name, subfolder) or ""
    if not folder_path or not os.path.isdir(folder_path):
        return pd.DataFrame()

    title_map = get_topic_title_map()
    topic_scores = defaultdict(list)

    # 1. Check for Random*.json files (Phase 1 or updated Phase 2 format)
    random_files = sorted([f for f in os.listdir(folder_path) if f.startswith("Random") and f.endswith(".json")])

    if random_files:
        for fname in random_files:
            fpath = os.path.join(folder_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = json.load(f)
                for raw_key, metrics in content.items():
                    if isinstance(metrics, dict) and "ndcg_cut_5" in metrics:
                        match = re.search(r"\d+$", raw_key)
                        if match:
                            t_key = f"T{int(match.group())}"
                            topic_scores[t_key].append(float(metrics["ndcg_cut_5"]))
            except Exception:
                pass
    else:
        # Fallback: check topics_values.json
        tv_path = os.path.join(folder_path, "topics_values.json")
        if os.path.exists(tv_path):
            try:
                with open(tv_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                for raw_key, vals in content.items():
                    if isinstance(vals, list) and vals:
                        match = re.search(r"\d+$", raw_key)
                        if match:
                            t_key = f"T{int(match.group())}"
                            topic_scores[t_key].extend([float(v) for v in vals])
            except Exception:
                pass

    if not topic_scores:
        return pd.DataFrame()

    rows = []
    for i in range(1, 46):
        t_key = f"T{i}"
        vals = topic_scores.get(t_key, [])
        if not vals:
            continue
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        min_val = float(min(vals))
        max_val = float(max(vals))
        range_val = float(max_val - min_val)
        title = title_map.get(t_key, t_key)

        rows.append({
            "Topic": t_key,
            "Topic Title": title,
            "Mean": round(mean_val, 4),
            "Std": round(std_val, 4),
            "Min": round(min_val, 4),
            "Max": round(max_val, 4),
            "Range": round(range_val, 4),
            "N_Seeds": len(vals),
            "Values": vals,
        })

    return pd.DataFrame(rows)