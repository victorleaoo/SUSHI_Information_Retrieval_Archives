"""
Q5 — Concept Bridge query generator.

Goal: Generate a multi-level list of related concepts and terms that bridge the
gap between the user's query vocabulary and the SNC/folder label vocabulary.

Why this works: The user searches "coffee" but the folder is labeled "AGRICULTURE".
Q5 adds "agriculture, agricultural commodities, coffee exports, crop production" —
terms that overlap with both the query intent and folder label vocabulary.

Post-processing: strip section labels, join all terms with spaces.
"""
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm_client import LLMClient

P1R_CONCEPT_TEMPLATE = """You are an expert on U.S. State Department archival classification. The archive uses Subject-Numeric Codes (SNC) to organize folders on Brazil (1960s-1970s).

Here are examples of REAL SNC category labels from the collection:
- AGRICULTURE
- POLITICAL AFFAIRS & RELATIONS
- LABOR & MANPOWER
- SCIENCE & TECHNOLOGY
- DEFENSE AFFAIRS
- ECONOMIC AFFAIRS
- HEALTH & SANITATION
- EDUCATION & CULTURE
- SOCIAL CONDITIONS
- HEAD OF STATE. EXECUTIVE BRANCH.
- GOVERNMENT: PROVINCIAL, MUNICIPAL & STATE GOVERNMENT
- INTERNAL SECURITY & INTELLIGENCE

Given the following research topic, generate a list of RELATED CONCEPTS AND TERMS that would help match this topic to the correct archival folders.

Research Topic Title: {title}
Research Topic Description: {description}

Generate terms at THREE levels:
1. BROAD CATEGORIES: Which general archival categories (like the examples above) would contain relevant folders? List 2-4.
2. SUB-TOPICS: What specific sub-topics within those categories relate to this query? List 5-8 terms.
3. CONCRETE TERMS: What specific terms, commodities, programs, or concepts would appear in documents filed under these folders? List 8-12 terms. Focus on terms relevant to Brazil in the 1960s-1970s.

Return three lines:
CATEGORIES: [comma-separated list]
SUB-TOPICS: [comma-separated list]
TERMS: [comma-separated list]

No explanations, no numbering beyond the three line labels."""


def _postprocess_concept(response: str) -> str:
    """Strips section labels and joins all terms into one query string."""
    combined = []
    for line in response.strip().splitlines():
        line = line.strip()
        for prefix in ('CATEGORIES:', 'SUB-TOPICS:', 'TERMS:'):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        # Add terms from this line
        for term in line.split(','):
            term = term.strip()
            if term:
                combined.append(term)
    return ' '.join(combined)


def generate_concept_queries(
    topics: list,
    llm: LLMClient,
    output_path: str,
    logs_path: str = None,
    topics_subset: list = None,
) -> dict:
    """Generate Q5 (Concept Bridge) queries for all 45 topics.

    Args:
        topics: List of topic dicts with keys ID, TITLE, DESCRIPTION.
        llm: Initialised LLMClient instance.
        output_path: Path to save the output JSON {topic_id: query_string}.
        logs_path: If set, append detailed JSONL log entries here.
        topics_subset: Optional list of topic IDs or ints to limit generation.

    Returns:
        Dict mapping topic_id → Q5 query string.
    """
    results = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception:
            pass

    # Filter topics
    filtered = topics
    if topics_subset is not None:
        if all(isinstance(x, int) for x in topics_subset):
            filtered = [t for t in topics if int(t['ID'].split('-')[-1]) in topics_subset]
        else:
            filtered = [t for t in topics if t['ID'] in topics_subset]

    # Skip already generated
    to_generate = [t for t in filtered if t['ID'] not in results]
    print(f"  [Q5-Concept] Generating for {len(to_generate)} topics "
          f"({len(filtered) - len(to_generate)} already cached)...")

    log_entries = []
    for topic in to_generate:
        prompt = P1R_CONCEPT_TEMPLATE.format(
            title=topic['TITLE'],
            description=topic['DESCRIPTION'],
        )
        print(f"    [Q5] → {topic['ID']}: {topic['TITLE'][:60]}")
        raw = llm.generate(prompt)
        q5 = _postprocess_concept(raw)
        results[topic['ID']] = q5
        print(f"    [Q5] ✓ Generated ({len(q5.split())} terms)")

        log_entries.append({
            'query_type': 'Q5_concept',
            'topic_id': topic['ID'],
            'topic_title': topic['TITLE'],
            'prompt': prompt,
            'raw_response': raw,
            'processed_query': q5,
        })

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Append to JSONL log
    if logs_path and log_entries:
        os.makedirs(os.path.dirname(os.path.abspath(logs_path)), exist_ok=True)
        with open(logs_path, 'a', encoding='utf-8') as f:
            for entry in log_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"  [Q5-Concept] Saved to {output_path}")
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
    generate_concept_queries(
        topics, llm,
        output_path=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'queries_concept.json'),
        logs_path=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'logs', 'phase1_queries', 'q5_concept.jsonl'),
    )
