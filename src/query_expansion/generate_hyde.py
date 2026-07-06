import json
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm_client import LLMClient

P1_HYDE_TEMPLATE = """You are an expert on U.S. State Department records from the 1960s and 1970s, specifically diplomatic archives on Brazil. The State Department organized records using Subject-Numeric Codes (SNC). Here are representative examples of SNC folder labels from the collection:

- POL 15-1 BRAZ 01/01/1964: Executive branch of the Brazilian government
- POL 23-8 BRAZ 01/01/1965: Riots and civil unrest in Brazil
- DEF 19-3 BRAZ 01/01/1967: Military equipment and arms transfers to Brazil
- ECON 6 BRAZ 01/01/1963: Economic and financial statistics for Brazil
- LAB 3 BRAZ 01/01/1966: Labor conditions and union activities in Brazil
- AID 9 BRAZ 01/01/1964: U.S. economic and military assistance to Brazil
- POL 7 BRAZ 01/01/1969: Visits and meetings between government officials
- MIL 6 BRAZ 01/01/1968: Military exercises and operations in Brazil

Given the following research topic, generate 1 to 3 SNC-style folder labels that would describe the folders most likely to contain relevant documents. It must contain the code, date and the SNC description.

Research Topic Title: {title}
Research Topic Description: {description}

Return ONLY the SNC-style labels, one per line. No explanations, no numbering, no preamble."""

def generate_hyde_queries(topics: list[dict], llm: LLMClient, output_path: str, topics_subset: list = None):
    """Generate Q1 (HyDE) queries for all 45 topics."""
    results = {}
    if os.path.exists(output_path):
        try:
            import json as _json_temp
            with open(output_path, 'r', encoding='utf-8') as f:
                results = _json_temp.load(f)
        except:
            pass

    filtered_topics = topics
    if topics_subset is not None:
        if all(isinstance(x, int) for x in topics_subset):
            filtered_topics = [t for t in topics if int(t['ID'].split('-')[-1]) in topics_subset]
        else:
            filtered_topics = [t for t in topics if t['ID'] in topics_subset]
    print(f"Generating HyDE queries (Q1) for {len(filtered_topics)} topics...")
    for topic in filtered_topics:
        prompt = P1_HYDE_TEMPLATE.format(
            title=topic['TITLE'],
            description=topic['DESCRIPTION']
        )
        response = llm.generate(prompt)
        # Post-processing: concatenate all generated labels
        q1 = ' '.join(response.strip().split('\n'))
        results[topic['ID']] = q1
        print(f"Generated Q1 for {topic['ID']}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    # Example usage
    topics_path = "../../src/data_creation/topics_output.txt"
    if not os.path.exists(topics_path):
        topics_path = "../data_creation/topics_output.txt"
    
    with open(topics_path, 'r', encoding='utf-8') as f:
        topics_data = json.load(f)
    
    topics = []
    for t_id, t_info in topics_data.items():
        t = t_info.copy()
        if 'ID' not in t:
            t['ID'] = t_id
        topics.append(t)
        
    llm = LLMClient(cache_dir='../../data/llm_cache', provider='groq')
    # You can specify a list of topic IDs or integers here (e.g., [12, 13])
    topics_subset = None
    generate_hyde_queries(topics, llm, '../../data/queries_hyde.json', topics_subset=topics_subset)
