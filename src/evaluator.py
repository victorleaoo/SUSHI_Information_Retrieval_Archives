import os
import json
import re
import math
import numpy as np
import scipy.stats as st
import pytrec_eval

class Evaluator:
    """
    Handles the evaluation of retrieval results against ground truth (QRELs).

    This class is responsible for:
    1. Saving search results in the standard TREC format.
    2. evaluating individual run files using `pytrec_eval` metrics (nDCG, MAP, etc.).
    3. Calculating custom metrics, such as the count of relevant folders in the Top 5.
    4. Aggregating results across multiple random seeds to produce statistical summaries (Mean/Margin).

    Attributes:
        folder_qrels_path (str): Path to the QREL file defining relevant folders.
        box_qrels_path (str): Path to the QREL file defining relevant boxes (unused in folder eval).
        measures (set): The set of pytrec_eval metrics to calculate (e.g., 'ndcg_cut_5').
    """
    def __init__(self, 
                 folder_qrels_path, 
                 box_qrels_path):
        self.folder_qrels_path = folder_qrels_path
        self.box_qrels_path = box_qrels_path
        self.measures = {'ndcg_cut', 'map', 'recip_rank', 'success'}

    def save_run_file(self, results, output_path, run_name):
        """
        Writes the search results to a standard TREC run file.

        The output format is: `query_id Q0 doc_id rank score run_name`. The score is derived from the rank (1/rank) as a proxy.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for topic in results:
                for i, doc_id in enumerate(topic['RankedList']):
                    # TREC format: query_id Q0 doc_id rank score run_name
                    print(f'{topic["Id"]}\t{doc_id}\t{i+1}\t{1/(i+1):.4f}\t{run_name}', file=f)

    def evaluate(self, run_file_path, output_json_path):
        """
        Evaluates a single run file against QRELs.

        This method performs two types of evaluation:
        1. Standard Information Retrieval metrics (nDCG@5, MAP, MRR) using `pytrec_eval`.
        2. A custom metric: `count_relevant_top5`, which counts how many relevant items appear strictly within the top 5 results.
        """
        # 1. Load Run
        with open(run_file_path) as runFile:
            run_data = {}
            run_lists = {}
            for line in runFile:
                topicId, folderId, _, score, _ = line.split('\t')
                if topicId not in run_data:
                    run_data[topicId] = {}
                    run_lists[topicId] = []
                run_data[topicId][folderId] = float(score)
                run_lists[topicId].append(folderId)

        # 2. Load QRELs
        with open(self.folder_qrels_path) as qrelsFile:
            qrels_data = {}
            for line in qrelsFile:
                topicId, _, folderId, relevanceLevel = line.split('\t')
                if topicId not in qrels_data:
                    qrels_data[topicId] = {}
                qrels_data[topicId][folderId] = int(relevanceLevel.strip())

        # 3. Evaluate
        evaluator = pytrec_eval.RelevanceEvaluator(qrels_data, self.measures)
        results = evaluator.evaluate(run_data)

        for topic_id in results:
            # Get the top 5 folders for this topic (sorted list)
            top_5_folders = run_lists.get(topic_id, [])[:5]
            
            relevant_count_top5 = 0
            topic_qrels = qrels_data.get(topic_id, {})
            
            for folder in top_5_folders:
                if folder in topic_qrels and topic_qrels[folder] > 0:
                    relevant_count_top5 += 1
            
            results[topic_id]['count_relevant_top5'] = relevant_count_top5

        # 4. Save
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)

    def generate_aggregated_metrics(self, folder_path, run_type):
        """
        Aggregates metrics across multiple random seed executions.

        Reads all individual metric files (e.g., 'Random42_TopicsFolderMetrics.json') 
        in the specified directory. It calculates:
        1. Mean and Margin (95% CI) for nDCG@5 per topic.
        2. Mean Count of Relevant Items (Top 5) per topic.
        3. Global Mean nDCG across all topics and seeds.

        Generates three output files in `folder_path`:
        - `topics_mean_margin.json`: nDCG statistics per topic.
        - `topics_relevant_count_stats.json`: Relevance count statistics per topic.
        - `model_overall_stats.json`: Global performance summary.
        """
        topic_accumulator = {f"T{i}": [] for i in range(1, 46)}
        count_accumulator = {f"T{i}": [] for i in range(1, 46)}
        
        # 1. Gather Data
        if run_type == 'random':
            valid_files = [f for f in os.listdir(folder_path) if f.startswith("Random") and f.endswith(".json")]
            for filename in valid_files:
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                run_values = {}
                for raw_key, metrics in data.items():
                    # Extract 1 from "T1" or similar patterns
                    norm_key = int(re.search(r'\d+$', raw_key).group())
                    run_values[norm_key] = metrics.get('ndcg_cut_5', 0.0)
                    count_accumulator[f"T{norm_key}"].append(metrics.get('count_relevant_top5', 0))
                    
                
                for topic in topic_accumulator:
                    t_id = int(topic[1:])
                    topic_accumulator[topic].append(run_values.get(t_id, 0.0))

        elif run_type == 'all_documents':
            # Logic for single run
            single_file = os.path.join(folder_path, "AllDocuments_TopicsFolderMetrics.json")
            if os.path.exists(single_file):
                with open(single_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                for k, v in raw_data.items():
                    t_id = int(re.search(r'\d+$', k).group())
                    key = f"T{t_id}"
                    if key in topic_accumulator:
                        val = v.get('ndcg_cut_5', 0.0)
                        topic_accumulator[key] = [val] # Single item list

        # 2. Calculate Stats
        topics_intervals = {}
        with open(os.path.join(folder_path, "topics_values.json"), 'w') as f:
            json.dump(topic_accumulator, f, indent=4)

        for topic, values in topic_accumulator.items():
            mean, margin = self._calculate_mean_margin(values)
            lower = max(0.0, mean - margin)
            topics_intervals[topic] = (lower, mean, mean + margin)
        
        with open(os.path.join(folder_path, "topics_mean_margin.json"), 'w') as f:
            json.dump(topics_intervals, f, indent=4)

        count_stats = {}
        for topic, values in count_accumulator.items(): 
            count_stats[topic] = {
                "mean": np.mean(values)
            }
            
        with open(os.path.join(folder_path, "topics_relevant_count_stats.json"), 'w') as f:
            json.dump(count_stats, f, indent=4)

        # 3. Global Stats
        all_topics_means = [val[1] for val in topics_intervals.values()]
        global_mean, global_margin = self._calculate_mean_margin(all_topics_means)
        
        model_stats = {
            "model_global_ndcg": {
                "mean": global_mean,
                "margin": global_margin,
                "interval": [max(0.0, global_mean - global_margin), global_mean, min(1.0, global_mean + global_margin)]
            }
        }
        
        print(model_stats)
        
        filename = "model_overall_stats.json" if run_type == 'random' else "all_documents_model_overall_stats.json"
        with open(os.path.join(folder_path, filename), 'w') as f:
            json.dump(model_stats, f, indent=4)

    def _calculate_mean_margin(self, values):
        """
        Helper method to calculate mean and 95% confidence interval margin.

        Returns:
            tuple: (mean, margin)
        """
        if not values: 
            return 0.0, 0.0
        n = len(values)
        mean = np.mean(values)
        if mean == 0.0 or n < 2: 
            return mean, 0.0
        
        sem = st.sem(values)
        interval = st.t.interval(0.95, df=n-1, loc=mean, scale=sem)
        margin = interval[1] - mean
        return mean, margin