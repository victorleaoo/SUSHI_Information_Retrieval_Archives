"""
Q2R — Term-Focused Keywords query generator.

Goal: Generate thematic search terms optimised for matching against FOLDER LABELS
(not document content). Replaces Q2 (Original Keywords).

Why this revises Q2: The original Q2 generates proper nouns (specific people,
specific institutions) — terms that rarely appear in folder labels. Q2R generates
thematic terms (land reform, military governance, agricultural policy) that match
the vocabulary used in SNC descriptions and folder labels. Also includes SNC
parent abbreviations (POL, AGR, DEF) as explicit bridge terms.

Post-processing: replace commas with spaces.
"""
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm_client import LLMClient

P1R_KW_TEMPLATE = """You are a research assistant specializing in archival classification systems. Generate a list of search terms optimized for matching against FOLDER LABELS (not document content) in a U.S. State Department archive on Brazil (1960s-1970s).

Research Topic Title: {title}
Research Topic Description: {description}

Focus your terms on:
- Thematic categories and sub-categories (e.g., "agriculture", "political affairs", "labor relations")
- Domain-specific concepts (e.g., "land reform", "military government", "economic development")
- Related terms and synonyms that broaden the match (e.g., for "coffee" → "crops, commodities, exports, agricultural production")
- Brazilian-context terms relevant to the 1960s-1970s
- Abbreviations used in State Department classification (e.g., POL, AGR, DEF, ECON)

DO NOT focus on:
- Specific people's names (these rarely appear in folder labels)
- Specific event dates
- Geographic locations (unless the topic is specifically about a location)

Return ONLY a single comma-separated list. Maximum 25 terms."""


def generate_keywords_r_queries(
    topics: list,
    llm: LLMClient,
    output_path: str,
    logs_path: str = None,
    topics_subset: list = None,
) -> dict:
    """Generate Q2R (Term-Focused Keywords) queries for all 45 topics.

    Args:
        topics: List of topic dicts with keys ID, TITLE, DESCRIPTION.
        llm: Initialised LLMClient instance.
        output_path: Path to save the output JSON {topic_id: query_string}.
        logs_path: If set, append detailed JSONL log entries here.
        topics_subset: Optional list of topic IDs or ints to limit generation.

    Returns:
        Dict mapping topic_id → Q2R query string.
    """
    results = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception:
            pass

    filtered = topics
    if topics_subset is not None:
        if all(isinstance(x, int) for x in topics_subset):
            filtered = [t for t in topics if int(t['ID'].split('-')[-1]) in topics_subset]
        else:
            filtered = [t for t in topics if t['ID'] in topics_subset]

    to_generate = [t for t in filtered if t['ID'] not in results]
    print(f"  [Q2R-Keywords-R] Generating for {len(to_generate)} topics "
          f"({len(filtered) - len(to_generate)} already cached)...")

    log_entries = []
    for topic in to_generate:
        prompt = P1R_KW_TEMPLATE.format(
            title=topic['TITLE'],
            description=topic['DESCRIPTION'],
        )
        print(f"    [Q2R] → {topic['ID']}: {topic['TITLE'][:60]}")
        raw = llm.generate(prompt)
        # Post-processing: replace commas with spaces, compact whitespace
        q2r = ' '.join(raw.replace(',', ' ').replace('\n', ' ').split())
        results[topic['ID']] = q2r
        print(f"    [Q2R] ✓ Generated ({len(q2r.split())} terms)")

        log_entries.append({
            'query_type': 'Q2R_keywords_r',
            'topic_id': topic['ID'],
            'topic_title': topic['TITLE'],
            'prompt': prompt,
            'raw_response': raw,
            'processed_query': q2r,
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if logs_path and log_entries:
        os.makedirs(os.path.dirname(os.path.abspath(logs_path)), exist_ok=True)
        with open(logs_path, 'a', encoding='utf-8') as f:
            for entry in log_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"  [Q2R-Keywords-R] Saved to {output_path}")
    return results


if __name__ == '__main__':
    topics_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'data_creation', 'topics_output.txt')
    with open(topics_path, 'r', encoding='utf-8') as f:
        topics_data = json.load(f)
    topics = [{'ID': tid, **info} for tid, info in topics_data.items()]

    llm = LLMClient(
        cache_dir=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'llm_cache'),
        provider='ollama',
        logs_dir=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'logs', 'llm_calls'),
    )
    generate_keywords_r_queries(
        topics, llm,
        output_path=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'queries_keywords_r.json'),
        logs_path=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'logs', 'phase1_queries', 'q2r_keywords_r.jsonl'),
    )
