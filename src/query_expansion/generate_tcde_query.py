import json
import os
import sys

# Add the 'src' directory to sys.path to allow importing 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm_client import LLMClient

def generate_tcde_queries(topics, llm_client: LLMClient, output_path: str):
    results = {}
    for topic in topics:
        tid = topic['ID']
        title = topic.get('TITLE', '')
        desc = topic.get('DESCRIPTION', '')
        
        prompt = f"""You are a research assistant specializing in U.S.-Brazil diplomatic archives (1960s-1970s). Given the research topic below, generate 3 diverse passages that each explore a different dimension of the same information need. Each passage should be 2-3 sentences and simulate the kind of description that might appear in an archival finding aid.

Research Topic Title: {title}
Research Topic Description: {desc}

Generate exactly 3 passages, each from a different angle: (1) institutional/policy, (2) actors/people, (3) events/actions.

Return the 3 passages separated by a blank line. No numbering, no labels."""
        
        response = llm_client.generate(prompt)
        passages = response.strip().split('\n\n')
        passages_text = " ".join([p.strip() for p in passages if p.strip()])
        
        # Post-processing: Q-TCDE = {title} {description} × 3 + all 3 passages concatenated
        base = f"{title} {desc}"
        q_tcde = f"{base} {base} {base} {passages_text}"
        results[tid] = q_tcde
        print(f"Generated Q-TCDE for {tid}")
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == '__main__':
    from utils.llm_client import LLMClient
    llm = LLMClient(cache_dir='../../data/llm_cache', provider='groq')
    with open('../../src/data_creation/topics_output.txt', 'r', encoding='utf-8') as f:
        topics_data = json.load(f)
    topics = [{'ID': tid, **info} for tid, info in topics_data.items()]
    generate_tcde_queries(topics, llm, '../../data/queries_tcde.json')
