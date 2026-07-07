import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm_client import LLMClient

P1_KW_TEMPLATE = """You are a research assistant specializing in U.S.-Brazil diplomatic history from the 1960s and 1970s. Generate a comprehensive list of search terms for retrieving relevant archival documents on the topic below.

Research Topic Title: {title}
Research Topic Description: {description}

Include in your list:
- Full names of specific people (politicians, diplomats, military officers, government officials relevant to this topic)
- Names of organizations and institutions (Brazilian government bodies, U.S. agencies, international bodies)
- Geographic locations specifically relevant to this topic
- Names of specific operations, programs, legislation, events, or incidents
- Technical terms and domain-specific jargon
- Common abbreviations used in State Department documents
- Key Portuguese-language names or terms that would appear in files

Return ONLY a single comma-separated list. No bullet points, no numbering, no explanations. Maximum 30 terms."""

def generate_keywords_queries(topics: list, llm: LLMClient, output_path: str,
                              topics_subset: list = None, logs_path: str = None) -> dict:
    """Generate Q2 (Keywords) queries for all 45 topics."""
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
    print(f"  [Q2-Keywords] Generating for {len(to_generate)} topics "
          f"({len(filtered_topics) - len(to_generate)} already cached)...")

    log_entries = []
    for topic in to_generate:
        prompt = P1_KW_TEMPLATE.format(
            title=topic['TITLE'],
            description=topic['DESCRIPTION']
        )
        print(f"    [Q2] → {topic['ID']}: {topic['TITLE'][:60]}")
        raw = llm.generate(prompt)
        q2 = ' '.join(raw.replace(',', ' ').replace('\n', ' ').split())
        results[topic['ID']] = q2
        print(f"    [Q2] ✓ Generated ({len(q2.split())} terms)")
        log_entries.append({'query_type': 'Q2_keywords', 'topic_id': topic['ID'],
                            'topic_title': topic['TITLE'], 'prompt': prompt,
                            'raw_response': raw, 'processed_query': q2})

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if logs_path and log_entries:
        os.makedirs(os.path.dirname(os.path.abspath(logs_path)), exist_ok=True)
        with open(logs_path, 'a', encoding='utf-8') as f:
            for entry in log_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"  [Q2-Keywords] Saved to {output_path}")
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
    generate_keywords_queries(topics, llm, '../../data/queries_keywords.json', topics_subset=topics_subset)
