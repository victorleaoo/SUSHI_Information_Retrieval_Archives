"""
Q6 — SNC-Vocabulary-Aware HyDE query generator.

Goal: Generate hypothetical folder descriptions using vocabulary drawn from
REAL SNC labels (not invented SNC codes). This replaces Q1 (Original HyDE).

Why this replaces Q1: The original HyDE generates fake SNC codes that don't
exist in the index. Q6 generates descriptions *in the style of real SNC labels*,
producing vocabulary that actually overlaps with indexed folder text.

Post-processing: join generated lines with spaces as Q6 string.
"""
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm_client import LLMClient

P1R_HYDE_TEMPLATE = """You are an expert on U.S. State Department archival classification for Brazil (1960s-1970s).
Folders in this archive are labeled with Subject-Numeric Codes (SNC) that describe their thematic content. Here are examples of REAL folder label descriptions:

- "AGRICULTURE" — covers crop production, agricultural policy, commodities
- "POLITICAL AFFAIRS & RELATIONS: HEAD OF STATE. EXECUTIVE BRANCH." — covers presidential actions, executive decisions
- "POLITICAL AFFAIRS & RELATIONS: GOVERNMENT: PROVINCIAL, MUNICIPAL & STATE GOVERNMENT" — covers regional politics
- "LABOR & MANPOWER: LABOR CONDITIONS & LABOR RELATIONS" — covers unions, strikes, working conditions
- "SCIENCE & TECHNOLOGY: TECHNOLOGICAL RESEARCH" — covers scientific programs, technology transfer
- "DEFENSE AFFAIRS: ARMED FORCES" — covers military operations, defense policy

Given the following research topic, write 2-3 SHORT folder descriptions (10-20 words each) in the SAME STYLE as the examples above. Describe what a matching folder would be about thematically — do NOT invent SNC codes or dates.

Research Topic Title: {title}
Research Topic Description: {description}

Return ONLY the folder descriptions, one per line. No codes, no dates, no explanations."""


def generate_hyde_r_queries(
    topics: list,
    llm: LLMClient,
    output_path: str,
    logs_path: str = None,
    topics_subset: list = None,
) -> dict:
    """Generate Q6 (SNC-Aware HyDE) queries for all 45 topics.

    Args:
        topics: List of topic dicts with keys ID, TITLE, DESCRIPTION.
        llm: Initialised LLMClient instance.
        output_path: Path to save the output JSON {topic_id: query_string}.
        logs_path: If set, append detailed JSONL log entries here.
        topics_subset: Optional list of topic IDs or ints to limit generation.

    Returns:
        Dict mapping topic_id → Q6 query string.
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
    print(f"  [Q6-HyDE-R] Generating for {len(to_generate)} topics "
          f"({len(filtered) - len(to_generate)} already cached)...")

    log_entries = []
    for topic in to_generate:
        prompt = P1R_HYDE_TEMPLATE.format(
            title=topic['TITLE'],
            description=topic['DESCRIPTION'],
        )
        print(f"    [Q6] → {topic['ID']}: {topic['TITLE'][:60]}")
        raw = llm.generate(prompt)
        # Post-processing: join lines with spaces
        q6 = ' '.join(line.strip().strip('"') for line in raw.strip().splitlines() if line.strip())
        results[topic['ID']] = q6
        print(f"    [Q6] ✓ Generated ({len(q6.split())} words)")

        log_entries.append({
            'query_type': 'Q6_hyde_r',
            'topic_id': topic['ID'],
            'topic_title': topic['TITLE'],
            'prompt': prompt,
            'raw_response': raw,
            'processed_query': q6,
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if logs_path and log_entries:
        os.makedirs(os.path.dirname(os.path.abspath(logs_path)), exist_ok=True)
        with open(logs_path, 'a', encoding='utf-8') as f:
            for entry in log_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"  [Q6-HyDE-R] Saved to {output_path}")
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
    generate_hyde_r_queries(
        topics, llm,
        output_path=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'queries_hyde_r.json'),
        logs_path=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'logs', 'phase1_queries', 'q6_hyde_r.jsonl'),
    )
