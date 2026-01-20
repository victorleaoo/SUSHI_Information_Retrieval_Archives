import os
import json
import re
import math
import numpy as np
import scipy.stats as st
import pytrec_eval

class Evaluator:
    def __init__(self, folder_qrels_path, box_qrels_path):
        self.folder_qrels_path = folder_qrels_path
        self.box_qrels_path = box_qrels_path
        self.measures = {'ndcg_cut', 'map', 'recip_rank', 'success'}

    def save_run_file(self, results, output_path, run_name):
        """Writes the standard TREC run file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for topic in results:
                for i, doc_id in enumerate(topic['RankedList']):
                    # TREC format: query_id Q0 doc_id rank score run_name
                    # utilizing 1/(i+1) as a proxy score for rank
                    print(f'{topic["Id"]}\t{doc_id}\t{i+1}\t{1/(i+1):.4f}\t{run_name}', file=f)

    def evaluate(self, run_file_path, output_json_path):
        """Compares the run file against QRELs using pytrec_eval."""
        # 1. Load Run
        with open(run_file_path) as runFile:
            run_data = {}
            for line in runFile:
                topicId, folderId, _, score, _ = line.split('\t')
                if topicId not in run_data:
                    run_data[topicId] = {}
                run_data[topicId][folderId] = float(score)

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

        # 4. Save
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)

    def generate_aggregated_metrics(self, folder_path, run_type):
        """
        Reads all JSON metrics in a folder and calculates Mean/Margin/Intervals.
        Replaces the old 'metric_file_generator'.
        """
        topic_accumulator = {f"T{i}": [] for i in range(1, 46)}
        
        # 1. Gather Data
        if run_type == 'random':
            valid_files = [f for f in os.listdir(folder_path) if f.startswith("Random") and f.endswith(".json")]
            for filename in valid_files:
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                run_values = {}
                for raw_key, metrics in data.items():
                    # Extract T1 from "T1" or similar patterns
                    norm_key = int(re.search(r'\d+$', raw_key).group())
                    run_values[norm_key] = metrics.get('ndcg_cut_5', 0.0)
                
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