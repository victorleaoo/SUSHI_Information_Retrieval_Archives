import numpy as np
import scipy.stats as st
import json
import os
import sys
import re

def normalize_topic_key(key: str) -> str:
    """Ensures keys like 'topic01' or '001' become 'T1'."""
    try:
        topic_num = int(re.search(r'\d+$', key).group())
        return f"T{topic_num}"
    except (AttributeError, ValueError):
        return key

def calculate_topic_confidence(ndcg_values, confidence=0.95):
    n = len(ndcg_values)

    mean = np.mean(ndcg_values)

    if mean == 0.0:
        return 0.0, 0.0

    # Erro Padrão da Média (Standard Error of Mean - SEM)
    # s / sqrt(n)
    sem = st.sem(ndcg_values) 
    
    # Intervalo usando T-Student
    # df = n - 1
    interval = st.t.interval(confidence, df=n-1, loc=mean, scale=sem)
    
    # A margem de erro é a distância da média até a borda do intervalo
    margin_of_error = interval[1] - mean
    
    return mean, margin_of_error

def calculate_model_consolidated_stats(topics_intervals_dict):
    """
    Calcula a estatística global do modelo baseada na variância entre tópicos.
    """
    # Extrai apenas a média (índice 1) de cada tópico
    # topics_intervals_dict[topic] = (lower, mean, upper)
    all_topics_means = [val[1] for val in topics_intervals_dict.values()]
    
    # Calcula o IC global considerando n = número de tópicos (45)
    global_mean, global_margin = calculate_topic_confidence(all_topics_means)
    
    lower = max(0.0, global_mean - global_margin)
    upper = global_mean + global_margin
    
    return lower, global_mean, upper

def load_json_safely(filepath: str):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def metric_generator(random_generation = True):
    paths = []
    file_values_path = 'topics_values.json' # GENERATED FROM acc -> calculate_folder_average
    if random_generation:
        for item in os.listdir('./all_runs/'):
            # Construct the full path
            full_path = os.path.join('./all_runs/', item)
            
            # Check if the item is a directory
            if os.path.isdir(full_path):
                paths.append(os.path.join(full_path, file_values_path))

                valid_files = [f for f in os.listdir(full_path) if f.startswith("Random") and f.endswith(".json")]

                topic_accumulator = {topic: [] for topic in {f"T{i}" for i in range(1, 46)}}
                for filename in valid_files:
                    with open(os.path.join(full_path, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    run_values = {}
                    for raw_key, metrics in data.items():
                        norm_key = normalize_topic_key(raw_key)
                        val = metrics.get('ndcg_cut_5', 0.0)
                        run_values[norm_key] = val
                    
                    # Add to accumulator (fill missing topics with 0.0 for this run)
                    for topic in {f"T{i}" for i in range(1, 46)}:
                        topic_accumulator[topic].append(run_values.get(topic, 0.0))

                with open(os.path.join(os.path.join(full_path, file_values_path)), 'w') as f:
                    json.dump(topic_accumulator, f, indent=4)

                #print(f"Saved all runs for: {os.path.join(os.path.join(full_path, file_values_path))}")
            else:
                continue
        
        for file_path in paths:
            with open(file_path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)

            topics_intervals = {}
            for topic, values in data.items():
                mean, margin = calculate_topic_confidence(values)
                lower = max(0.0, mean - margin)
                topics_intervals[topic] = (lower, mean, mean + margin)

            parent_dir = os.path.dirname(file_path)
            
            output_topics_path = os.path.join(parent_dir, "topics_mean_margin.json")
            with open(output_topics_path, 'w') as f:
                json.dump(topics_intervals, f, indent=4)

            model_lower, model_mean, model_upper = calculate_model_consolidated_stats(topics_intervals)
            
            model_stats = {
                "model_global_ndcg": {
                    "mean": model_mean,
                    "margin": model_upper - model_mean,
                    "interval": [model_lower, model_mean, model_upper]
                }
            }

            output_model_path = os.path.join(parent_dir, "model_overall_stats.json")
            with open(output_model_path, 'w') as f:
                json.dump(model_stats, f, indent=4)
    else:
        for item in os.listdir('./all_runs/'):
            # Construct the full path
            full_path = os.path.join('../all_runs/', item)
            
            prefixo_arquivo = "AllTrainingDocuments"
            if os.path.isdir(full_path):
                for filename in os.listdir(full_path):
                    if filename.startswith(prefixo_arquivo):
                        paths.append(os.path.join(full_path, filename))
        
        expected_topics = {f"T{i}" for i in range(1, 46)}
        
        for file_path in paths:
            with open(file_path, 'r', encoding='utf-8') as handle:
                raw_data = json.load(handle)
                
                data = {}
                for k, v in raw_data.items():
                    data[normalize_topic_key(k)] = v

            topics_intervals = {}
            
            # Itera obrigatoriamente de T1 a T45
            for topic in expected_topics:
                if topic in data:
                    val = data[topic]['ndcg_cut_5']
                    topics_intervals[topic] = (val, val, val)
                else:
                    # Se o tópico não existir no arquivo, assume 0.0
                    topics_intervals[topic] = (0.0, 0.0, 0.0)

            parent_dir = os.path.dirname(file_path)

            model_lower, model_mean, model_upper = calculate_model_consolidated_stats(topics_intervals)
            
            model_stats = {
                "model_global_ndcg": {
                    "mean": model_mean,
                    "margin": model_upper - model_mean,
                    "interval": [model_lower, model_mean, model_upper]
                }
            }

            output_model_path = os.path.join(parent_dir, "all_training_model_overall_stats.json")
            with open(output_model_path, 'w') as f:
                json.dump(model_stats, f, indent=4)