"""
Q7 — Thematic Paragraph query generator.

Goal: Generate a rich thematic paragraph focused on concepts and terms rather
than specific events and figures. Replaces Q3 (Embedding Text).

Why this revises Q3: Original Q3 over-focuses on specific historical events and
diplomatic actors. Q7 emphasises thematic concepts, related terminology, and
the types of content that would be in matching folders — better for semantic
matching against archival folder labels.

Post-processing: use the paragraph directly as Q7 string.
"""
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm_client import LLMClient

P1R_THEMATIC_TEMPLATE = """You are an expert on U.S.-Brazil diplomatic archives (1960s-1970s). Write a thematic description for the research topic below, optimized for matching against archival folder labels and descriptions.

Research Topic Title: {title}
Research Topic Description: {description}

Write a single paragraph of 100-150 words that:
- Describes the THEMATIC SCOPE of what this topic covers (not specific events)
- Lists related concepts, sub-topics, and domain terms that a researcher would associate with this topic
- Uses vocabulary that would appear in archival folder labels and classification systems
- Mentions the types of content (policy documents, reports, cables, assessments) that would be filed under this topic
- Includes related concepts that broaden the match (e.g., for "coffee" also mention "agricultural policy, commodity exports, trade agreements")

Focus on TERMS AND CONCEPTS, not on narrating specific events or naming specific people.

Write only the paragraph. No title, no header, no preamble."""


def generate_thematic_queries(
    topics: list,
    llm: LLMClient,
    output_path: str,
    logs_path: str = None,
    topics_subset: list = None,
) -> dict:
    """Generate Q7 (Thematic Paragraph) queries for all 45 topics.

    Args:
        topics: List of topic dicts with keys ID, TITLE, DESCRIPTION.
        llm: Initialised LLMClient instance.
        output_path: Path to save the output JSON {topic_id: query_string}.
        logs_path: If set, append detailed JSONL log entries here.
        topics_subset: Optional list of topic IDs or ints to limit generation.

    Returns:
        Dict mapping topic_id → Q7 query string.
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
    print(f"  [Q7-Thematic] Generating for {len(to_generate)} topics "
          f"({len(filtered) - len(to_generate)} already cached)...")

    log_entries = []
    for topic in to_generate:
        prompt = P1R_THEMATIC_TEMPLATE.format(
            title=topic['TITLE'],
            description=topic['DESCRIPTION'],
        )
        print(f"    [Q7] → {topic['ID']}: {topic['TITLE'][:60]}")
        raw = llm.generate(prompt)
        # Post-processing: use paragraph directly, collapse newlines
        q7 = ' '.join(raw.strip().split())
        results[topic['ID']] = q7
        print(f"    [Q7] ✓ Generated ({len(q7.split())} words)")

        log_entries.append({
            'query_type': 'Q7_thematic',
            'topic_id': topic['ID'],
            'topic_title': topic['TITLE'],
            'prompt': prompt,
            'raw_response': raw,
            'processed_query': q7,
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if logs_path and log_entries:
        os.makedirs(os.path.dirname(os.path.abspath(logs_path)), exist_ok=True)
        with open(logs_path, 'a', encoding='utf-8') as f:
            for entry in log_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"  [Q7-Thematic] Saved to {output_path}")
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
    generate_thematic_queries(
        topics, llm,
        output_path=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'queries_thematic.json'),
        logs_path=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'logs', 'phase1_queries', 'q7_thematic.jsonl'),
    )
