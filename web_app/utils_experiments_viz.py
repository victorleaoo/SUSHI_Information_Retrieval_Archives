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
    'Avg With Expansion': '#1f77b4',
    'Avg No Expansion': '#ff7f0e',
    'AllTrainingDocs': '#2ca02c',
    'SUSHISubmissions': '#d62728',
    'Embeddings (F_EMB_T)': '#9467bd'
}

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

def get_smart_folder_name(search_field: str, expansion: bool, query_field: str) -> str:
    exp_str = "EX" if expansion else "NEX"
    return f"{search_field}_{exp_str}_{query_field}"

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
    css_rules = []
    for idx, col_name in enumerate(df.columns):
        css_index = idx + 1
        topic_num = get_topic_number(col_name)
        if topic_num != -1:
            if topic_num in DIFFICULT_TOPICS:
                rule = (f".dataframe td:nth-child({css_index}), "
                        f".dataframe th:nth-child({css_index}) "
                        f"{{ background-color: #e3bf88 !important; }}")
                css_rules.append(rule)
            elif topic_num in IMPOSSIBLE_TOPICS:
                rule = (f".dataframe td:nth-child({css_index}), "
                        f".dataframe th:nth-child({css_index}) "
                        f"{{ background-color: #ab5050 !important; }}")
                css_rules.append(rule)
    return "\n".join(css_rules)

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

def load_oracle_data(folder_path: str) -> Dict[str, Dict]:
    if not os.path.exists(folder_path): return {}
    clean_data = {f"T{i}": {'ndcg_cut_5': 0.0} for i in range(1, 46)}
    for filename in os.listdir(folder_path):
        if filename.startswith("AllTrainingDocuments"):
            data = load_json_safely(os.path.join(folder_path, filename))
            if not data: continue
            for key, val in data.items():
                norm_key = normalize_topic_key(key)
                clean_data[norm_key] = val
            return clean_data
    return clean_data

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
        if oracle_data and t in oracle_data:
            val = oracle_data[t].get('ndcg_cut_5', 0)
            chart_rows.append({"Topic": t, "Type": "AllTrainingDocs", "nDCG": val, "min_ci": val, "max_ci": val})
        if sushi_data and t in sushi_data:
            val = sushi_data[t].get('ndcg_cut_5', 0)
            chart_rows.append({"Topic": t, "Type": "SUSHISubmissions", "nDCG": val, "min_ci": val, "max_ci": val})
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

def process_experiment_data(search_field: str, query_field: str):
    name_ex = get_smart_folder_name(search_field, True, query_field)
    name_nex = get_smart_folder_name(search_field, False, query_field)
    path_ex = os.path.join(EXPERIMENTS_ROOT_DIR, name_ex)
    path_nex = os.path.join(EXPERIMENTS_ROOT_DIR, name_nex)
    
    avg_ex, files_ex = calculate_folder_average(path_ex)
    avg_nex, files_nex = calculate_folder_average(path_nex)
    margins_ex = load_margin_data(path_ex)
    margins_nex = load_margin_data(path_nex)
    stats_ex = load_overall_stats(path_ex, "model_overall_stats.json")
    stats_nex = load_overall_stats(path_nex, "model_overall_stats.json")
    
    stats_oracle = load_overall_stats(path_nex, "all_training_model_overall_stats.json")
    oracle_data = load_oracle_data(path_nex)
    sushi_data = load_sushi_submissions(SUSHI_ROOT_DIR)

    final_stats_ex = get_model_metric_summary(stats_ex, avg_ex)
    final_stats_nex = get_model_metric_summary(stats_nex, avg_nex)
    final_stats_oracle = get_model_metric_summary(stats_oracle, oracle_data)

    all_topics = set()
    if avg_ex: all_topics.update(avg_ex.keys())
    if avg_nex: all_topics.update(avg_nex.keys())
    if oracle_data: all_topics.update(oracle_data.keys())
    sorted_topics = sorted(list(all_topics), key=natural_keys)

    is_embedding_scenario = (search_field == 'F' and query_field == 'T')
    path_emb = os.path.join(EXPERIMENTS_ROOT_DIR, "F_EMB_T")
    avg_emb, margins_emb, stats_emb = {}, {}, {}
    
    if is_embedding_scenario:
        avg_emb, _ = calculate_folder_average(path_emb)
        margins_emb = load_margin_data(path_emb)
        raw_stats = load_overall_stats(path_emb, "model_overall_stats.json")
        stats_emb = get_model_metric_summary(raw_stats, avg_emb)

    df_chart = build_chart_dataset(sorted_topics, avg_ex, margins_ex, avg_nex, margins_nex, oracle_data, sushi_data, avg_emb, margins_emb)
    df_table = build_table_dataset(sorted_topics, avg_ex, avg_nex, oracle_data, sushi_data, path_ex, files_ex, path_nex, files_nex)

    return df_chart, df_table, sorted_topics, final_stats_ex, final_stats_nex, final_stats_oracle, stats_emb