import os
from tqdm import tqdm

from run_generator import RunGenerator, Style, RANDOM_SEED_LIST, RESULTS_PATH

# ==========================================
# HYBRID RRF LOGIC
# ==========================================
def perform_hybrid_fusion(results_a, results_b, k=0, weight_a=1.0, weight_b=0.65):
    """
    Combines two lists of results using Weighted Reciprocal Rank Fusion.
    
    Args:
        results_a (list): List of results for model A.
        results_b (list): List of results for model B.
        k (int): RRF constant (default 0).
        weight_a (float): Weight for results_a (default 1.0).
        weight_b (float): Weight for results_b (default 0.65).

    Returns:
        list: The merged results.
    """
    merged_results = []
    
    # Create lookup for Set B
    dict_b = {item['Id']: item['RankedList'] for item in results_b}

    for item_a in results_a:
        topic_id = item_a['Id']
        list_a = item_a['RankedList']
        list_b = dict_b.get(topic_id, [])

        scores = {}
        
        # Helper to add scores with a specific weight
        def add_scores(doc_list, weight):
            for rank, doc in enumerate(doc_list):
                # RRF Formula: weight * (1 / (k + rank))
                # rank is 0-indexed, so we use rank + 1
                score = weight * (1 / (k + (rank + 1)))
                scores[doc] = scores.get(doc, 0.0) + score

        # Apply weighting here
        add_scores(list_a, weight_a) # Weight 1.0
        add_scores(list_b, weight_b) # Weight 0.65

        # Sort by accumulated score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        final_list = [doc for doc, score in sorted_docs]
        
        merged_results.append({'Id': topic_id, 'RankedList': final_list})
        
    return merged_results

def run_hybrid_experiment():
    """
    Runs the hybrid RRF experiment.
    """
    print(f"{Style.BOLD}{Style.GREEN}> Starting Hybrid RRF Experiment (TOFS_NEX_TD_BM25-EMBEDDINGS-COLBERT-TUNED + ALLFL_NEX_TD_COLBERT){Style.RESET}")

    # --- CONFIGURATION 1: TOFS (Document-based) ---
    search_field_A = ['title', 'ocr', 'folderlabel', 'summary']
    gen_A = RunGenerator(
        searching_fields=[search_field_A],
        query_fields=['TD'],
        models=['bm25', 'embeddings', 'colbert'],
        # Expansion Config for Doc Model
        expansion=['similar_snc'], 
        rrf_input='docs', # Early fusion usually better for expansion
        expansion_ceiling_k=1,
        all_folders_folder_label=False
    )

    # --- CONFIGURATION 2: ALLFL (Folder-based) ---
    search_field_B = ['folderlabel'] 
    gen_B = RunGenerator(
        searching_fields=[search_field_B],
        query_fields=['TD'],
        models=['colbert'],
        expansion=[], # No expansion for folder-based
        all_folders_folder_label=True
    )
    
    # --- OUTPUT SETUP ---
    run_folder_name = "HYBRID-TOFS-SMS-1-ALLFL-COLBERT_NE_TD_BM25-EMBEDDINGS-COLBERT-TUNED-WRRF"
    metrics_output_folder = os.path.abspath(f'../all_runs/{run_folder_name}')
    os.makedirs(metrics_output_folder, exist_ok=True)

    # --- MAIN LOOP ---
    for random_seed in tqdm(RANDOM_SEED_LIST, desc="Hybrid RRF Runs"):
        # 1. Run Config A
        results_A = gen_A.run_single_seed(random_seed, search_field_A, 'TD')

        # 2. Run Config B
        results_B = gen_B.run_single_seed(random_seed, search_field_B, 'TD')

        # 3. Fuse Results (RRF)
        final_results = perform_hybrid_fusion(results_A, results_B)

        # 4. Save & Evaluate
        run_name = f'45-Topics-Random-{random_seed}'
        gen_A.evaluator.save_run_file(final_results, RESULTS_PATH, run_name)
        
        json_path = os.path.join(metrics_output_folder, f'Random{random_seed}_TopicsFolderMetrics.json')
        gen_A.evaluator.evaluate(RESULTS_PATH, json_path)

    # 5. Aggregate
    print(f"> Generating Aggregated Metrics in {metrics_output_folder}...")
    gen_A.evaluator.generate_aggregated_metrics(metrics_output_folder, 'random')
    print(f"{Style.BOLD}{Style.GREEN}> Hybrid Experiment Complete!{Style.RESET}")

if __name__ == "__main__":
    run_hybrid_experiment()