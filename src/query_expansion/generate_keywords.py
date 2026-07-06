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

def generate_keywords_queries(topics: list[dict], llm: LLMClient, output_path: str, topics_subset: list = None):
    """Generate Q2 (Keywords) queries for all 45 topics."""
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
    print(f"Generating Keywords queries (Q2) for {len(filtered_topics)} topics...")
    for topic in filtered_topics:
        prompt = P1_KW_TEMPLATE.format(
            title=topic['TITLE'],
            description=topic['DESCRIPTION']
        )
        response = llm.generate(prompt)
        # Post-processing: Replace commas with spaces
        q2 = response.replace(',', ' ').replace('\n', ' ')
        # compact multiple spaces
        q2 = ' '.join(q2.split())
        results[topic['ID']] = q2
        print(f"Generated Q2 for {topic['ID']}")
    
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
    generate_keywords_queries(topics, llm, '../../data/queries_keywords.json', topics_subset=topics_subset)
