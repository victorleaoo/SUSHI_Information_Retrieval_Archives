import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm_client import LLMClient

P1_REF_TEMPLATE = """You are an expert in archival research on U.S. State Department records on Brazil (1960s-1970s). Your task is to expand the following research topic by generating 4 NOVEL and DISTINCT search queries. DO NOT just copy or slightly reword the provided title and description. You must use your historical knowledge to infer related concepts, specific names, and events that are not explicitly mentioned but are highly relevant to the topic.

Research Topic Title: {title}
Research Topic Description: {description}

Generate 4 queries, each approaching the topic from a different angle:
Query 1 (U.S. Perspective): Focus on the official U.S. government perspective, actions, policies, or specific agencies involved.
Query 2 (Brazilian Context): Focus on Brazilian actors, institutions, domestic politics, or local context.
Query 3 (Specific Events/Policies): Focus on specific historical events, operations, incidents, or policies related to the topic.
Query 4 (Key Entities): Focus on key named individuals, groups, or organizations.

Rules:
- CRITICAL: Do NOT simply repeat phrases from the title or description. Generate NEW vocabulary and terms related to the topic.
- Each query must be 5-15 words long.
- Maximize the diversity of vocabulary across the 4 queries.
- Return ONLY the 4 queries, one per line, without numbering, labels, or explanations."""

def generate_reformulations_queries(topics: list, llm: LLMClient, output_path: str,
                                    topics_subset: list = None, logs_path: str = None) -> dict:
    """Generate Q4 (Reformulations) queries for all 45 topics."""
    results = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception:
            pass

    filtered_topics = topics
    if topics_subset is not None:
        if all(isinstance(x, int) for x in topics_subset):
            filtered_topics = [t for t in topics if int(t['ID'].split('-')[-1]) in topics_subset]
        else:
            filtered_topics = [t for t in topics if t['ID'] in topics_subset]

    to_generate = [t for t in filtered_topics if t['ID'] not in results]
    print(f"  [Q4-Reformulations] Generating for {len(to_generate)} topics "
          f"({len(filtered_topics) - len(to_generate)} already cached)...")

    log_entries = []
    for topic in to_generate:
        prompt = P1_REF_TEMPLATE.format(
            title=topic['TITLE'],
            description=topic['DESCRIPTION']
        )
        print(f"    [Q4] → {topic['ID']}: {topic['TITLE'][:60]}")
        raw = llm.generate(prompt)
        q4 = ' '.join(raw.strip().split('\n'))
        results[topic['ID']] = q4
        print(f"    [Q4] ✓ Generated")
        log_entries.append({'query_type': 'Q4_reformulations', 'topic_id': topic['ID'],
                            'topic_title': topic['TITLE'], 'prompt': prompt,
                            'raw_response': raw, 'processed_query': q4})

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if logs_path and log_entries:
        os.makedirs(os.path.dirname(os.path.abspath(logs_path)), exist_ok=True)
        with open(logs_path, 'a', encoding='utf-8') as f:
            for entry in log_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"  [Q4-Reformulations] Saved to {output_path}")
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
    generate_reformulations_queries(topics, llm, '../../data/queries_reformulations.json', topics_subset=topics_subset)
