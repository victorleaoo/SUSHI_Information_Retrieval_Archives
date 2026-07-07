"""
Phase 1 — Query Generation Script.

Generates ALL query types for Phase 1 experiments:
  Original:  Q1 (HyDE), Q2 (Keywords), Q3 (EmbText), Q4 (Reformulations)
  Revised:   Q5 (Concept Bridge), Q6 (SNC-Aware HyDE), Q7 (Thematic), Q2R (Term KW)
  Combined:  Q0+Qx, QALL, Q0+QALL, Q0+QALL-R, etc.

Usage:
  python scripts/run_phase1_generate_queries.py [--provider ollama|groq|openai] [--model MODEL]

Run this ONCE before any 1A/1B/1BR experiment scripts.
Queries are cached on disk — re-running is safe and only generates missing queries.

Logs are written to:
  data/logs/llm_calls/          — per-call JSON logs (prompt + response)
  data/logs/phase1_queries/     — per-query-type JSONL logs
"""
import sys
import os
import json
import argparse
from datetime import datetime

# Make src importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from utils.llm_client import LLMClient
from query_expansion.generate_hyde import generate_hyde_queries
from query_expansion.generate_keywords import generate_keywords_queries
from query_expansion.generate_emb_text import generate_emb_text_queries
from query_expansion.generate_reformulations import generate_reformulations_queries
from query_expansion.generate_concept import generate_concept_queries
from query_expansion.generate_hyde_r import generate_hyde_r_queries
from query_expansion.generate_thematic import generate_thematic_queries
from query_expansion.generate_keywords_r import generate_keywords_r_queries
from query_expansion.combine_queries import combine_queries
from query_expansion.combine_queries_r import combine_queries_r

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
TOPICS_PATH  = os.path.join(PROJECT_ROOT, 'src', 'data_creation', 'topics_output.txt')
LOGS_DIR     = os.path.join(DATA_DIR, 'logs')


def _log_phase(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"\n{'='*60}")
    print(f"[{ts}] {msg}")
    print('='*60)


def load_topics() -> list:
    _log_phase("Loading topics from topics_output.txt")
    with open(TOPICS_PATH, 'r', encoding='utf-8') as f:
        topics_data = json.load(f)
    topics = [{'ID': tid, **info} for tid, info in topics_data.items()]
    print(f"  Loaded {len(topics)} topics.")
    return topics


def main():
    parser = argparse.ArgumentParser(description='Generate all Phase 1 query expansions.')
    parser.add_argument('--provider', default='ollama', choices=['ollama', 'groq', 'openai'],
                        help='LLM provider to use (default: ollama)')
    parser.add_argument('--model', default=None,
                        help='Model name override (default: provider default)')
    parser.add_argument('--ollama-url', default='http://localhost:11434',
                        help='Ollama base URL (default: http://localhost:11434)')
    args = parser.parse_args()

    run_start = datetime.now()
    _log_phase(f"Phase 1 — Query Generation Started")
    print(f"  Provider: {args.provider}")
    print(f"  Model:    {args.model or '(provider default)'}")
    print(f"  Data dir: {DATA_DIR}")

    # ── LLM Client ─────────────────────────────────────────────────────────────
    llm = LLMClient(
        cache_dir=os.path.join(DATA_DIR, 'llm_cache'),
        provider=args.provider,
        model=args.model,
        logs_dir=os.path.join(LOGS_DIR, 'llm_calls'),
        ollama_url=args.ollama_url,
    )

    topics = load_topics()
    q_logs_dir = os.path.join(LOGS_DIR, 'phase1_queries')
    os.makedirs(q_logs_dir, exist_ok=True)

    # ── Original Query Generation ───────────────────────────────────────────────
    _log_phase("Step 1/9 — Generating Q1 (HyDE, Original)")
    q1_path = os.path.join(DATA_DIR, 'queries_hyde.json')
    generate_hyde_queries(topics, llm, q1_path,
                          logs_path=os.path.join(q_logs_dir, 'q1_hyde.jsonl'))

    _log_phase("Step 2/9 — Generating Q2 (Keywords, Original)")
    q2_path = os.path.join(DATA_DIR, 'queries_keywords.json')
    generate_keywords_queries(topics, llm, q2_path,
                              logs_path=os.path.join(q_logs_dir, 'q2_keywords.jsonl'))

    _log_phase("Step 3/9 — Generating Q3 (Embedding Text, Original)")
    q3_path = os.path.join(DATA_DIR, 'queries_embtext.json')
    generate_emb_text_queries(topics, llm, q3_path,
                              logs_path=os.path.join(q_logs_dir, 'q3_embtext.jsonl'))

    _log_phase("Step 4/9 — Generating Q4 (Reformulations, Original)")
    q4_path = os.path.join(DATA_DIR, 'queries_reformulations.json')
    generate_reformulations_queries(topics, llm, q4_path,
                                    logs_path=os.path.join(q_logs_dir, 'q4_reformulations.jsonl'))

    _log_phase("Step 5/9 — Combining Original Queries (Q0+Qx, QALL, Q0+QALL)")
    combine_queries(q1_path, q2_path, q3_path, q4_path, TOPICS_PATH, DATA_DIR)

    # ── Revised Query Generation ────────────────────────────────────────────────
    _log_phase("Step 6/9 — Generating Q5 (Concept Bridge, Revised)")
    q5_path = os.path.join(DATA_DIR, 'queries_concept.json')
    generate_concept_queries(topics, llm, q5_path,
                             logs_path=os.path.join(q_logs_dir, 'q5_concept.jsonl'))

    _log_phase("Step 7/9 — Generating Q6 (SNC-Aware HyDE, Revised)")
    q6_path = os.path.join(DATA_DIR, 'queries_hyde_r.json')
    generate_hyde_r_queries(topics, llm, q6_path,
                            logs_path=os.path.join(q_logs_dir, 'q6_hyde_r.jsonl'))

    _log_phase("Step 8/9 — Generating Q7 (Thematic Paragraph, Revised)")
    q7_path = os.path.join(DATA_DIR, 'queries_thematic.json')
    generate_thematic_queries(topics, llm, q7_path,
                              logs_path=os.path.join(q_logs_dir, 'q7_thematic.jsonl'))

    _log_phase("Step 9/9 — Generating Q2R (Term-Focused Keywords, Revised)")
    q2r_path = os.path.join(DATA_DIR, 'queries_keywords_r.json')
    generate_keywords_r_queries(topics, llm, q2r_path,
                                logs_path=os.path.join(q_logs_dir, 'q2r_keywords_r.jsonl'))

    _log_phase("Step 9b/9 — Combining Revised Queries (Q0+Q5, Q0+Q6, Q0+QALL-R, ...)")
    combine_queries_r(q5_path, q6_path, q7_path, q2r_path, TOPICS_PATH, DATA_DIR)

    # ── Summary ─────────────────────────────────────────────────────────────────
    elapsed = (datetime.now() - run_start).total_seconds()
    _log_phase(f"Query Generation Complete ✓")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Query files saved to: {DATA_DIR}")
    print(f"  LLM call logs:        {os.path.join(LOGS_DIR, 'llm_calls')}")
    print(f"  Query JSONL logs:     {q_logs_dir}")
    print()
    print("  Files generated:")
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.startswith('queries_') and fname.endswith('.json'):
            fpath = os.path.join(DATA_DIR, fname)
            size = os.path.getsize(fpath) // 1024
            print(f"    {fname:45s}  {size:>4} KB")


if __name__ == '__main__':
    main()
