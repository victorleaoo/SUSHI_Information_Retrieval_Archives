import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm_client import LLMClient

P1_EMB_TEMPLATE = """You are a diplomatic historian specializing in U.S.-Brazil relations during the 1960s and 1970s. Write a detailed descriptive paragraph for the research topic below, as if writing for an academic encyclopedia on U.S. diplomatic archives.

Research Topic Title: {title}
Research Topic Description: {description}

Write a single paragraph of 150-200 words that:
- Describes the topic with historical specificity for the Brazil-U.S. context
- Mentions key actors, institutions, and events relevant to this topic during the 1960s-1970s
- Uses the terminology and phrasing characteristic of State Department documents
- Captures the political and diplomatic context a researcher would need
- Focuses on what kinds of archival records would satisfy this information need

Write only the paragraph. No title, no header, no preamble."""

def generate_emb_text_queries(topics: list[dict], llm: LLMClient, output_path: str, topics_subset: list = None):
    """Generate Q3 (Embedding Text) queries for all 45 topics."""
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
    print(f"Generating Embedding Text queries (Q3) for {len(filtered_topics)} topics...")
    for topic in filtered_topics:
        prompt = P1_EMB_TEMPLATE.format(
            title=topic['TITLE'],
            description=topic['DESCRIPTION']
        )
        response = llm.generate(prompt)
        # Post-processing: Use paragraph directly
        q3 = response.strip()
        results[topic['ID']] = q3
        print(f"Generated Q3 for {topic['ID']}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
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
    generate_emb_text_queries(topics, llm, '../../data/queries_embtext.json', topics_subset=topics_subset)
