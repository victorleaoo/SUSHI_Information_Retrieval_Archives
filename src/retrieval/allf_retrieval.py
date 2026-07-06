import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hybrid_models import perform_hybrid_fusion

def run_allf_retrieval(query_dict: dict, models_dict: dict, topics: list) -> list:
    """
    Runs retrieval for all topics given a query dict mapping topic_id to query string.
    models_dict is a dict mapping 'b', 'c', 'e' to their trained models (e.g. {'b': bm25_model}).
    Returns a list of dicts: [{'Id': topic_id, 'RankedList': [folder_id1, folder_id2, ...]}, ...]
    """
    
    # We will compute results for each base model that is present
    base_results = {}
    
    for m_id, model in models_dict.items():
        m_results = []
        for topic in topics:
            topic_id = topic['ID']
            query_str = query_dict.get(topic_id, '')
            
            # The model's search method returns a DataFrame with 'docno'
            df = model.search(query_str)
            # Some models might return score, but perform_hybrid_fusion only uses rank.
            # We just need the ordered list of docnos (which are folder_ids here).
            ranked_list = df['docno'].tolist() if not df.empty and 'docno' in df.columns else []
            m_results.append({'Id': topic_id, 'RankedList': ranked_list})
        base_results[m_id] = m_results
        
    # If there's only one model, return its results
    if len(models_dict) == 1:
        return list(base_results.values())[0]
        
    # Ensemble logic
    # The prompt mentions BCE: RRF(B + C + E, k=60), BC, BE, CE
    # Since perform_hybrid_fusion merges two lists, we can chain it.
    # Weights from experiments.md or hybrid_models: bm25: 1.0, embeddings: 0.65, colbert: 0.65
    
    weights = {'b': 1.0, 'e': 0.65, 'c': 0.65}
    k_val = 60
    
    if 'b' in base_results and 'c' in base_results and 'e' not in base_results: # BC
        return perform_hybrid_fusion(base_results['b'], base_results['c'], k=k_val, weight_a=weights['b'], weight_b=weights['c'])
    elif 'b' in base_results and 'e' in base_results and 'c' not in base_results: # BE
        return perform_hybrid_fusion(base_results['b'], base_results['e'], k=k_val, weight_a=weights['b'], weight_b=weights['e'])
    elif 'c' in base_results and 'e' in base_results and 'b' not in base_results: # CE
        return perform_hybrid_fusion(base_results['c'], base_results['e'], k=k_val, weight_a=weights['c'], weight_b=weights['e'])
    elif 'b' in base_results and 'c' in base_results and 'e' in base_results: # BCE
        # Merge B and C first
        bc_results = perform_hybrid_fusion(base_results['b'], base_results['c'], k=k_val, weight_a=weights['b'], weight_b=weights['c'])
        # In the second merge, bc_results already has weighted scores. We want to treat it as 'a' with weight 1.0, 
        # and 'e' as 'b' with weight 0.65. Wait, RRF merges lists. perform_hybrid_fusion only looks at rank of lists!
        # So merging lists repeatedly works but the weights apply to the RRF formula: weight * 1/(k+rank)
        # Actually, perform_hybrid_fusion applies RRF on the provided lists.
        # Let's write a custom multi-way RRF for BCE to be accurate.
        
        merged_results = []
        for i in range(len(topics)):
            topic_id = topics[i]['ID']
            scores = {}
            def add_scores(doc_list, weight):
                for rank, doc in enumerate(doc_list):
                    score = weight * (1 / (k_val + (rank + 1)))
                    scores[doc] = scores.get(doc, 0.0) + score
            
            add_scores(base_results['b'][i]['RankedList'], weights['b'])
            add_scores(base_results['c'][i]['RankedList'], weights['c'])
            add_scores(base_results['e'][i]['RankedList'], weights['e'])
            
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            final_list = [doc for doc, score in sorted_docs]
            merged_results.append({'Id': topic_id, 'RankedList': final_list})
            
        return merged_results
        
    return []
