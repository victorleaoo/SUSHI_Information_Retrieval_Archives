"""
Combines revised query outputs (Q5, Q6, Q7, Q2R) into composite query strings
for Set 1B-R experiments.

Combinations produced:
  QALL-R       = Q5 + Q6 + Q7 + Q2R        (LLM-only, no Q0)
  Q0+Q5        = Q0 + Q5
  Q0+Q6        = Q0 + Q6
  Q0+Q5+Q6     = Q0 + Q5 + Q6
  Q0+QALL-R    = Q0 + Q5 + Q6 + Q7 + Q2R  (most comprehensive revised)

Also saves a human-readable overview JSON for inspection.
"""
import json
import os


def combine_queries_r(
    q5_path: str,
    q6_path: str,
    q7_path: str,
    q2r_path: str,
    topics_path: str,
    output_dir: str,
) -> None:
    """
    Combines revised query files into composite query strings.

    Args:
        q5_path:     Path to queries_concept.json (Q5).
        q6_path:     Path to queries_hyde_r.json (Q6).
        q7_path:     Path to queries_thematic.json (Q7).
        q2r_path:    Path to queries_keywords_r.json (Q2R).
        topics_path: Path to topics_output.txt (for Q0 base strings).
        output_dir:  Directory to write combined query files.
    """
    print("[combine_queries_r] Loading revised query files...")
    with open(q5_path, 'r', encoding='utf-8') as f:
        q5 = json.load(f)
    with open(q6_path, 'r', encoding='utf-8') as f:
        q6 = json.load(f)
    with open(q7_path, 'r', encoding='utf-8') as f:
        q7 = json.load(f)
    with open(q2r_path, 'r', encoding='utf-8') as f:
        q2r = json.load(f)
    with open(topics_path, 'r', encoding='utf-8') as f:
        topics = json.load(f)

    qall_r = {}
    q0q5 = {}
    q0q6 = {}
    q0q5q6 = {}
    q0qall_r = {}
    overview_r = {}

    all_ids = set(q5) & set(q6) & set(q7) & set(q2r)
    missing = set(q5) ^ all_ids
    if missing:
        print(f"  Warning: {len(missing)} topic(s) missing in some revised files, skipping: {missing}")

    for tid in sorted(all_ids):
        topic_info = topics.get(tid, {})
        q0 = f"{topic_info.get('TITLE', '')} {topic_info.get('DESCRIPTION', '')}".strip()

        qall_r[tid]    = f"{q5[tid]} {q6[tid]} {q7[tid]} {q2r[tid]}"
        q0q5[tid]      = f"{q0} {q5[tid]}"
        q0q6[tid]      = f"{q0} {q6[tid]}"
        q0q5q6[tid]    = f"{q0} {q5[tid]} {q6[tid]}"
        q0qall_r[tid]  = f"{q0} {q5[tid]} {q6[tid]} {q7[tid]} {q2r[tid]}"

        overview_r[tid] = {
            'TITLE':       topic_info.get('TITLE', ''),
            'DESCRIPTION': topic_info.get('DESCRIPTION', ''),
            'Q0':          q0,
            'Q5_Concept':  q5[tid],
            'Q6_HyDE_R':   q6[tid],
            'Q7_Thematic': q7[tid],
            'Q2R_Keywords_R': q2r[tid],
        }

    os.makedirs(output_dir, exist_ok=True)

    files = {
        'queries_qall_r.json':    qall_r,
        'queries_q0q5.json':      q0q5,
        'queries_q0q6.json':      q0q6,
        'queries_q0q5q6.json':    q0q5q6,
        'queries_q0qall_r.json':  q0qall_r,
        'queries_overview_r.json': overview_r,
    }

    for filename, data in files.items():
        out_path = os.path.join(output_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [combine_queries_r] → {filename} ({len(data)} topics)")

    print(f"[combine_queries_r] All revised combined queries saved to {output_dir}")


if __name__ == '__main__':
    base = os.path.join(os.path.dirname(__file__), '..', '..')
    combine_queries_r(
        q5_path=os.path.join(base, 'data', 'queries_concept.json'),
        q6_path=os.path.join(base, 'data', 'queries_hyde_r.json'),
        q7_path=os.path.join(base, 'data', 'queries_thematic.json'),
        q2r_path=os.path.join(base, 'data', 'queries_keywords_r.json'),
        topics_path=os.path.join(base, 'src', 'data_creation', 'topics_output.txt'),
        output_dir=os.path.join(base, 'data'),
    )
