import json
import os

def combine_queries(q1_path, q2_path, q3_path, q4_path, topics_path, output_dir):
    """Combines saved outputs → Q0+Qx augmented queries, QALL, Q0+QALL, and overview."""
    
    with open(q1_path, 'r') as f: q1 = json.load(f)
    with open(q2_path, 'r') as f: q2 = json.load(f)
    with open(q3_path, 'r') as f: q3 = json.load(f)
    with open(q4_path, 'r') as f: q4 = json.load(f)
    with open(topics_path, 'r') as f: topics = json.load(f)
    
    # LLM-only combinations
    qall = {}
    
    # Q0-augmented individual expansions
    q0q1, q0q2, q0q3, q0q4 = {}, {}, {}, {}
    
    # Q0-augmented combined expansions
    q0qall = {}
    
    # Overview for human inspection
    overview = {}
    
    for tid in q1:
        if tid not in q2 or tid not in q3 or tid not in q4:
            print(f"Skipping {tid} due to missing query.")
            continue
        
        topic_info = topics.get(tid, {})
        q0 = f"{topic_info.get('TITLE', '')} {topic_info.get('DESCRIPTION', '')}"
        
        # LLM-only: all 4 expansions concatenated (no Q0)
        qall[tid] = f"{q1[tid]} {q2[tid]} {q3[tid]} {q4[tid]}"
        
        # Q0 + individual LLM expansion
        q0q1[tid] = f"{q0} {q1[tid]}"
        q0q2[tid] = f"{q0} {q2[tid]}"
        q0q3[tid] = f"{q0} {q3[tid]}"
        q0q4[tid] = f"{q0} {q4[tid]}"
        
        # Q0 + all LLM expansions
        q0qall[tid] = f"{q0} {q1[tid]} {q2[tid]} {q3[tid]} {q4[tid]}"
        
        # Human-readable overview
        overview[tid] = {
            "TITLE": topic_info.get("TITLE", ""),
            "DESCRIPTION": topic_info.get("DESCRIPTION", ""),
            "Q0": q0,
            "Q1_HyDE": q1[tid],
            "Q2_Keywords": q2[tid],
            "Q3_EmbText": q3[tid],
            "Q4_Reformulations": q4[tid]
        }
    
    os.makedirs(output_dir, exist_ok=True)
    
    files = {
        'queries_qall.json': qall,
        'queries_q0q1.json': q0q1,
        'queries_q0q2.json': q0q2,
        'queries_q0q3.json': q0q3,
        'queries_q0q4.json': q0q4,
        'queries_q0qall.json': q0qall,
        'queries_overview.json': overview
    }
    
    for filename, data in files.items():
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"Combined queries and overview saved to {output_dir}")

if __name__ == "__main__":
    combine_queries(
        '../../data/queries_hyde.json',
        '../../data/queries_keywords.json',
        '../../data/queries_embtext.json',
        '../../data/queries_reformulations.json',
        '../../src/data_creation/topics_output.txt',
        '../../data/'
    )
