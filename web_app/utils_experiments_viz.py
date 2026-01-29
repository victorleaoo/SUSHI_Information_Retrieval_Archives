import os
import json
import pandas as pd
import re
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

# ==========================================
# CONSTANTES
# ==========================================
EXPERIMENTS_ROOT_DIR = "../all_runs"
SUSHI_ROOT_DIR = "../all_runs/"

DIFFICULT_TOPICS = {24, 32, 39, 33, 27, 38, 42, 45, 41, 29, 40, 34, 9, 12, 6}
IMPOSSIBLE_TOPICS = {3, 8, 10, 13, 14, 17, 25, 26, 30, 31, 43}
ALL_KNOWN_TOPICS = {f"T{i}" for i in range(1, 46)}

COLOR_MAP = {
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
    parts = folder_name.split('_')
    if len(parts) != 4:
        return None
    
    return {
        "search": parts[0],
        "expansion": parts[1],
        "query": parts[2],
        "model": parts[3],
        "full_name": folder_name
    }

def scan_available_runs() -> Tuple[Dict[str, List[str]], List[Dict]]:
    """
    Scans the run directory and returns unique options for filters.
    """
    if not os.path.exists(EXPERIMENTS_ROOT_DIR):
        return {}, []

    options = {
        "search": set(),
        "query": set(),
        "model": set(),
        "expansion": set()
    }
    parsed_runs = []

    for f in os.listdir(EXPERIMENTS_ROOT_DIR):
        if not os.path.isdir(os.path.join(EXPERIMENTS_ROOT_DIR, f)):
            continue
            
        meta = parse_run_folder(f)
        if meta:
            options["search"].add(meta["search"])
            options["query"].add(meta["query"])
            options["model"].add(meta["model"])
            if meta["expansion"] != "NEX":
                options["expansion"].add(meta["expansion"])
            parsed_runs.append(meta)

    # Convert sets to sorted lists
    final_options = {k: sorted(list(v)) for k, v in options.items()}
    return final_options, parsed_runs

def natural_keys(text: str) -> List[Any]:
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def get_topic_number(col_name: str) -> int:
    try:
        if col_name.startswith("T"):
            return int(col_name[1:])
    except (ValueError, IndexError):
        pass
    return -1

def normalize_topic_key(key: str) -> str:
    try:
        topic_num = int(re.search(r'\d+$', key).group())
        return f"T{topic_num}"
    except (AttributeError, ValueError):
        return key

def get_smart_folder_name(search: str, expansion: str, query: str, model: str) -> str:
    """Reconstructs folder name based on strict 4-part convention."""
    return f"{search}_{expansion}_{query}_{model}"

def format_cell_content(metrics: Any) -> str:
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

def generate_column_css(df: pd.DataFrame) -> str:
    """
    Returns empty CSS. 
    Previous logic for coloring Difficult/Impossible topics has been removed 
    to keep the table uniform.
    """
    return ""

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
            
            # If value is missing (NaN), we might want to skip or handle gracefully
            # Your requirement: "repeat mean value if NaN". 
            # Note: avg_dict usually only contains valid keys.
            
            if t in avg_dict:
                val = avg_dict[t]
                # Default margin tuple is (val, val, val) -> no error bar
                m_min, _, m_max = margin_dict.get(t, [val, val, val])
                
                chart_rows.append({
                    "Topic": t, 
                    "Type": model_name, 
                    "nDCG": val, 
                    "min_ci": m_min, 
                    "max_ci": m_max
                })
                
    return pd.DataFrame(chart_rows)

# ==========================================
# DATA LOADING
# ==========================================

def load_json_safely(filepath: str) -> Dict:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def calculate_folder_average(folder_path: str) -> Tuple[Optional[Dict[str, float]], List[str]]:
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
            
    averages = {k: np.mean(v) for k, v in topic_accumulator.items() if v}
    return averages, valid_files

def load_margin_data(folder_path: str) -> Dict:
    return load_json_safely(os.path.join(folder_path, "topics_mean_margin.json"))

def load_overall_stats(folder_path: str, filename: str) -> Dict:
    data = load_json_safely(os.path.join(folder_path, filename))
    return data.get('model_global_ndcg', {})

def load_sushi_submissions(folder_path: str) -> Dict[str, Any]:
    if not os.path.exists(folder_path): return {}
    for filename in os.listdir(folder_path):
        if filename.startswith("SUSHISubmissions") and filename.endswith(".json"):
            data = load_json_safely(os.path.join(folder_path, filename))
            normalized_data = {}
            for key, val in data.items():
                normalized_data[normalize_topic_key(key)] = val
            return normalized_data
    return {}

def get_model_metric_summary(stats_dict: Dict, topic_avgs_dict: Dict) -> Dict[str, float]:
    if stats_dict and 'mean' in stats_dict:
        return {'val': stats_dict['mean'], 'margin': stats_dict.get('margin', 0.0)}
    elif topic_avgs_dict:
        return {'val': np.mean(list(topic_avgs_dict.values())), 'margin': 0.0}
    else:
        return {'val': 0.0, 'margin': 0.0}

def build_chart_dataset(topics: List[str], avg_ex: Dict, margins_ex: Dict, avg_nex: Dict, margins_nex: Dict, oracle_data: Dict, sushi_data: Dict, avg_emb: Dict = {}, margins_emb: Dict = {}) -> pd.DataFrame:
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

def build_table_dataset(sorted_topics: List[str], avg_ex: Dict, avg_nex: Dict, oracle_data: Dict, sushi_data: Dict, path_ex: str, files_ex: List[str], path_nex: str, files_nex: List[str]) -> pd.DataFrame:
    table_rows = []
    def add_row(name: str, source_data: Dict, is_raw: bool = False):
        row_data = {"Experiment": name}
        for t in sorted_topics:
            val = source_data.get(t, {}) if is_raw else source_data.get(t, 0.0)
            row_data[t] = format_cell_content(val)
        table_rows.append(row_data)
        
    def add_individual_files(folder_path: str, file_list: List[str], suffix_label: str):
        if not os.path.exists(folder_path): return
        for f in file_list:
            data = load_json_safely(os.path.join(folder_path, f))
            if data:
                clean = {normalize_topic_key(k): v for k, v in data.items()}
                simple_name = f.split('_')[0]
                add_row(f"{simple_name} - {suffix_label}", clean, is_raw=True)
                
    if avg_ex: add_row("Mean With Exp", avg_ex)
    if avg_nex: add_row("Mean No Exp", avg_nex)
    if oracle_data: add_row("All Training Docs", oracle_data, is_raw=True)
    if sushi_data: add_row("SUSHISubmissions", sushi_data, is_raw=True)
    
    add_individual_files(path_ex, files_ex, "With Expansion")
    add_individual_files(path_nex, files_nex, "Without Expansion")
    
    df_table = pd.DataFrame(table_rows)
    if not df_table.empty:
        cols = ["Experiment"] + sorted_topics
        df_table = df_table.reindex(columns=cols)
    return df_table

def get_grouped_run_configurations() -> Dict[str, List[str]]:
    """
    Scans the run directory and groups folders by their configuration prefix.
    Returns: { 'F_SB_TD': ['F_SB_TD_BM25', 'F_SB_TD_COLBERT'], ... }
    """
    if not os.path.exists(EXPERIMENTS_ROOT_DIR):
        return {}

    grouped_runs = {}

    for folder_name in os.listdir(EXPERIMENTS_ROOT_DIR):
        if not os.path.isdir(os.path.join(EXPERIMENTS_ROOT_DIR, folder_name)):
            continue
            
        # Naming Convention: Search_Expansion_Query_Model
        parts = folder_name.split('_')
        
        # We assume the LAST part is the Model, and the rest is Configuration
        if len(parts) >= 2:
            config_key = "_".join(parts[:-1]) # Everything before the last underscore
            
            if config_key not in grouped_runs:
                grouped_runs[config_key] = []
            
            grouped_runs[config_key].append(folder_name)

    return grouped_runs

def process_experiment_data(selected_configs: List[str], run_map: Dict[str, List[str]]):
    """
    Loads data for all models belonging to the selected configurations.
    """
    available_runs = []
    
    # 1. Gather all relevant folders
    for config in selected_configs:
        folders = run_map.get(config, [])
        for f in folders:
            meta = parse_run_folder(f)
            if meta:
                # We save the model name specifically for the legend
                meta['display_model'] = meta['model'] 
                available_runs.append(meta)

    # 2. Load Data
    model_results = {} 
    all_topics = set()
    
    for run in available_runs:
        path = os.path.join(EXPERIMENTS_ROOT_DIR, run['full_name'])
        
        # KEY CHANGE: The display key is just the Model Name (e.g. "BM25")
        display_name = run['display_model']
        
        avg_data, _ = calculate_folder_average(path)
        margin_data = load_margin_data(path)
        global_stats = load_overall_stats(path, "model_overall_stats.json")
        summary_stats = get_model_metric_summary(global_stats, avg_data)
        
        random_count = len([f for f in os.listdir(path) if f.startswith("Random") and f.endswith(".json") and "TopicsFolderMetrics" in f])

        model_results[display_name] = {
            'avg': avg_data,
            'margins': margin_data,
            'stats': summary_stats,
            'count': random_count,
            'folder_name': run['full_name']
        }
        
        if avg_data:
            all_topics.update(avg_data.keys())

    sorted_topics = sorted(list(all_topics), key=natural_keys)

    # 3. Build Output Objects
    df_chart = build_multi_model_chart_dataset(sorted_topics, model_results)
    
    # Simple table logic (unused in main view but good for internal consistency)
    table_rows = []
    for name, data in model_results.items():
        row = {'Experiment': name}
        avgs = data.get('avg', {})
        for t in sorted_topics:
            row[t] = format_cell_content(avgs.get(t, 0.0))
        table_rows.append(row)
        
    df_table = pd.DataFrame(table_rows)
    if not df_table.empty:
        df_table = df_table.reindex(columns=['Experiment'] + sorted_topics)

    return df_chart, df_table, sorted_topics, model_results

def get_all_runs_statistics() -> pd.DataFrame:
    """
    Scans the EXPERIMENTS_ROOT_DIR for all subfolders, looks for 
    model_overall_stats.json, and counts Random_ files.
    """
    rows = []
    
    if not os.path.exists(EXPERIMENTS_ROOT_DIR):
        return pd.DataFrame()

    # Iterate over all items in the root directory
    for folder_name in os.listdir(EXPERIMENTS_ROOT_DIR):
        folder_path = os.path.join(EXPERIMENTS_ROOT_DIR, folder_name)
        
        # Skip if it's not a directory
        if not os.path.isdir(folder_path):
            continue

        # Check for model_overall_stats.json
        stats_file_path = os.path.join(folder_path, "model_overall_stats.json")
        
        if os.path.exists(stats_file_path):
            stats_data = load_json_safely(stats_file_path)
            # Access the nested structure: "model_global_ndcg" -> "mean" / "margin"
            global_metrics = stats_data.get("model_global_ndcg", {})
            
            mean_val = global_metrics.get("mean", 0.0)
            margin_val = global_metrics.get("margin", 0.0)
            
            # Count files starting with "Random" (matches Random_ or Random1 etc)
            # Using startsWith("Random") based on your previous logic, 
            # but specific to prompt "Random_" can be adjusted if files are named strictly "Random_"
            random_count = len([
                f for f in os.listdir(folder_path) 
                if f.startswith("Random") and f.endswith(".json") and "TopicsFolderMetrics" in f
            ])
            
            rows.append({
                "Run Name": folder_name,
                "Mean": mean_val,
                "Margin": margin_val,
                "Random Runs Count": random_count
            })

    # Create DataFrame
    df = pd.DataFrame(rows)
    
    if not df.empty:
        # Sort by Mean descending by default
        df = df.sort_values(by="Run Name", ascending=False)
        
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
    
    # Pre-calculate sorted topic list to ensure column order
    all_topics = sorted(list(ALL_KNOWN_TOPICS), key=natural_keys)

    for run_name in selected_run_names:
        folder_path = os.path.join(EXPERIMENTS_ROOT_DIR, run_name)
        if not os.path.exists(folder_path): continue

        # 1. Load Data Sources
        # Global nDCG (Contains mean AND margin)
        overall_stats = load_overall_stats(folder_path, "model_overall_stats.json")
        
        # Per-Topic nDCG (Format: 'T1': [min, mean, max])
        topic_ndcg = load_margin_data(folder_path) 
        
        # Per-Topic Relevance (Format: 'T1': {'mean': X, ...})
        topic_rel = load_relevance_stats(folder_path)

        # 2. Process Global Stats
        # nDCG: We use the statistical mean and margin provided by the evaluator
        g_ndcg_mean = overall_stats.get('mean', 0.0)
        g_ndcg_margin = overall_stats.get('margin', 0.0)
        
        # Relevance: We calculate the simple average of the counts
        g_rel_mean = calculate_global_relevance_mean(topic_rel)

        # 3. Build Row
        row = {
            "Experiment Folder": run_name,
            "Global nDCG@5": f"{g_ndcg_mean:.4f} ± {g_ndcg_margin:.4f}",
            "Global Relevance": f"{g_rel_mean:.2f}" # No margin, just the number
        }

        # 4. Fill Topic Columns
        for t in all_topics:
            # Get nDCG Mean (Index 1 in the [min, mean, max] list)
            # We handle cases where data might be missing for a specific topic
            if topic_ndcg and t in topic_ndcg:
                ndcg_val = topic_ndcg[t][1]
                ndcg_margin = topic_ndcg[t][2] - topic_ndcg[t][1]
            else:
                ndcg_val = 0.0
                ndcg_margin = 0
            
            # Get Relevance Mean
            if topic_rel and t in topic_rel:
                rel_val = topic_rel[t].get('mean', 0.0)
            else:
                rel_val = 0.0
            
            # Format: "0.450 | 1.5"
            row[t] = f"{ndcg_val:.3f} ± {ndcg_margin:.3f} | {rel_val:.1f}"

        rows.append(row)

    # Create DF and sort
    df = pd.DataFrame(rows)
    if not df.empty:
        # Reorder columns: Fixed Info -> Metrics -> Topics
        cols = ["Experiment Folder", "Global nDCG@5", "Global Relevance"] + all_topics
        # Only keep columns that actually exist (in case no topics found)
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
    return df