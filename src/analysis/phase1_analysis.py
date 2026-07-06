import os
import json
import pandas as pd

def parse_phase1_results(all_runs_dir):
    """Parses Phase 1 run folders and returns DataFrames for 1A and 1B."""
    
    # Mapping for 1B runs: run_id → query type
    set_1b_runs = {
        # 1B-1: Q1 alone
        '1B-01': 'Q1', '1B-02': 'Q1', '1B-03': 'Q1', '1B-04': 'Q1', '1B-05': 'Q1', '1B-06': 'Q1', '1B-07': 'Q1',
        # 1B-2: Q2 alone
        '1B-08': 'Q2', '1B-09': 'Q2', '1B-10': 'Q2',
        # 1B-3: Q3 alone
        '1B-11': 'Q3', '1B-12': 'Q3', '1B-13': 'Q3', '1B-14': 'Q3',
        # 1B-4: Q4 alone
        '1B-15': 'Q4', '1B-16': 'Q4', '1B-17': 'Q4', '1B-18': 'Q4', '1B-19': 'Q4', '1B-20': 'Q4', '1B-21': 'Q4',
        # 1B-5: Q0 + individual LLM expansion
        '1B-22': 'Q0+Q1', '1B-23': 'Q0+Q1', '1B-24': 'Q0+Q1',
        '1B-25': 'Q0+Q2', '1B-26': 'Q0+Q2',
        '1B-27': 'Q0+Q3', '1B-28': 'Q0+Q3',
        '1B-29': 'Q0+Q4', '1B-30': 'Q0+Q4', '1B-31': 'Q0+Q4',
        # 1B-6: Combined expansions
        '1B-32': 'QALL', '1B-33': 'Q0+QALL', '1B-34': 'Q0+QALL', '1B-35': 'Q0+QALL',
        # 1B-7: Query-Index interactions
        '1B-36': 'Q0+QALL', '1B-37': 'Q0+QALL', '1B-38': 'Q0+QALL', '1B-39': 'Q0+QALL', '1B-40': 'Q0+QALL'
    }

    data_1a = []
    data_1b = []
    
    if not os.path.exists(all_runs_dir):
        print(f"Directory {all_runs_dir} not found.")
        return pd.DataFrame(), pd.DataFrame()

    for folder in os.listdir(all_runs_dir):
        if not folder.startswith("Phase1_"): continue
        
        parts = folder.split('_')
        if len(parts) != 4: continue
        
        _, run_id, config, model = parts
        
        stats_path = os.path.join(all_runs_dir, folder, 'all_documents_model_overall_stats.json')
        if not os.path.exists(stats_path):
            continue
            
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            ndcg5 = stats.get('model_global_ndcg', {}).get('mean', 0.0)
            
        if run_id.startswith('1A-'):
            data_1a.append({
                'RunID': run_id,
                'Config': config,
                'Model': model.upper(),
                'nDCG@5': ndcg5
            })
        elif run_id.startswith('1B-'):
            q_type = set_1b_runs.get(run_id, 'UNKNOWN')
            data_1b.append({
                'RunID': run_id,
                'QueryType': q_type,
                'Model': model.upper(),
                'Config': config,
                'nDCG@5': ndcg5
            })

    df_1a = pd.DataFrame(data_1a)
    df_1b = pd.DataFrame(data_1b)
    return df_1a, df_1b

def generate_markdown_report(df_1a, df_1b, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'phase1_report.md')
    
    models_order = ['B', 'E', 'C', 'BC', 'BE', 'CE', 'BCE']
    
    with open(report_path, 'w') as f:
        f.write("# Phase 1: All Folders Index Experiments Analysis\n\n")
        f.write("This report provides a tabular comparison of all runs from Phase 1. ")
        f.write("It helps visually compare 1A runs amongst themselves, 1B runs amongst themselves, and against each other.\n\n")
        
        # ── Phase 1A ──
        f.write("## Phase 1A: Base Queries (Q0) across Configurations\n")
        f.write("The table below shows **nDCG@5** for different index configurations (F1-F5) across all retrieval models.\n\n")
        if not df_1a.empty:
            pivot_1a = df_1a.pivot(index="Model", columns="Config", values="nDCG@5")
            pivot_1a = pivot_1a.reindex(index=[m for m in models_order if m in pivot_1a.index])
            f.write(pivot_1a.to_markdown(floatfmt=".4f"))
            f.write("\n\n")
            
            f.write("### Best Configuration per Model (Baseline for 1B)\n")
            best_idx = df_1a.groupby('Model')['nDCG@5'].idxmax()
            best_1a = df_1a.loc[best_idx].sort_values('nDCG@5', ascending=False)
            f.write(best_1a[['Model', 'Config', 'nDCG@5']].to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
        else:
            f.write("*No Phase 1A data found.*\n\n")
            
        # ── Phase 1B: LLM-only vs Q0+LLM ──
        if not df_1b.empty:
            # Split into LLM-only (Q1-Q4, QALL) and Q0-augmented (Q0+Qx)
            llm_only_types = ['Q1', 'Q2', 'Q3', 'Q4', 'QALL']
            q0_aug_types = ['Q0+Q1', 'Q0+Q2', 'Q0+Q3', 'Q0+Q4', 'Q0+QALL']
            
            df_llm = df_1b[df_1b['QueryType'].isin(llm_only_types)]
            df_q0 = df_1b[df_1b['QueryType'].isin(q0_aug_types)]
            
            # ── LLM-only table ──
            f.write("## Phase 1B (1-4): LLM Expansions Alone\n")
            f.write("These runs replace Q0 entirely with LLM-generated text.\n\n")
            if not df_llm.empty:
                pivot_llm = df_llm.pivot_table(index="Model", columns="QueryType", values="nDCG@5", aggfunc='first')
                pivot_llm = pivot_llm.reindex(index=[m for m in models_order if m in pivot_llm.index])
                
                if not df_1a.empty:
                    best_1a_series = df_1a.loc[df_1a.groupby('Model')['nDCG@5'].idxmax()].set_index('Model')[['nDCG@5']].rename(columns={'nDCG@5': 'Q0 (Baseline)'})
                    pivot_llm = pivot_llm.join(best_1a_series)
                    cols = ['Q0 (Baseline)'] + [c for c in pivot_llm.columns if c != 'Q0 (Baseline)']
                    pivot_llm = pivot_llm[cols]
                
                f.write(pivot_llm.to_markdown(floatfmt=".4f"))
                f.write("\n\n")
            
            # ── Q0-augmented table ──
            f.write("## Phase 1B (5-6): Q0-Augmented Queries\n")
            f.write("These runs preserve the original query (Q0) and augment it with LLM-generated text.\n\n")
            if not df_q0.empty:
                pivot_q0 = df_q0.pivot_table(index="Model", columns="QueryType", values="nDCG@5", aggfunc='first')
                pivot_q0 = pivot_q0.reindex(index=[m for m in models_order if m in pivot_q0.index])
                
                if not df_1a.empty:
                    best_1a_series = df_1a.loc[df_1a.groupby('Model')['nDCG@5'].idxmax()].set_index('Model')[['nDCG@5']].rename(columns={'nDCG@5': 'Q0 (Baseline)'})
                    pivot_q0 = pivot_q0.join(best_1a_series)
                    cols = ['Q0 (Baseline)'] + [c for c in pivot_q0.columns if c != 'Q0 (Baseline)']
                    pivot_q0 = pivot_q0[cols]
                
                f.write(pivot_q0.to_markdown(floatfmt=".4f"))
                f.write("\n\n")
            
            # ── Query-Index Interactions (1B-7) ──
            df_qi = df_1b[df_1b['RunID'].isin(['1B-36', '1B-37', '1B-38', '1B-39', '1B-40'])]
            if not df_qi.empty:
                f.write("## Phase 1B (7): Query-Index Interactions (Q0+QALL × Configs)\n")
                f.write("Tests whether richer index metadata interacts with the comprehensive augmented query.\n\n")
                qi_table = df_qi[['RunID', 'Config', 'nDCG@5']].sort_values('RunID')
                f.write(qi_table.to_markdown(index=False, floatfmt=".4f"))
                f.write("\n\n")
            
            # ── Overall Leaderboard ──
            f.write("## 🏆 Overall Top 15 Configurations (1A + 1B combined)\n")
            combined = pd.concat([
                df_1a[['Model', 'Config', 'nDCG@5']].assign(QueryType='Q0'),
                df_1b[['Model', 'Config', 'nDCG@5', 'QueryType']]
            ])
            top_n = combined.sort_values('nDCG@5', ascending=False).head(15)
            top_n = top_n[['Model', 'QueryType', 'Config', 'nDCG@5']]
            top_n.insert(0, 'Rank', range(1, len(top_n) + 1))
            f.write(top_n.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
        else:
            f.write("## Phase 1B\n*No Phase 1B data found.*\n\n")
            
    print(f"Report generated successfully at: {report_path}")

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    all_runs_dir = os.path.join(root_dir, 'all_runs')
    output_dir = os.path.join(root_dir, 'results', 'phase1_analysis')
    
    print("Parsing results...")
    df_1a, df_1b = parse_phase1_results(all_runs_dir)
    
    print(f"Found {len(df_1a)} runs for 1A and {len(df_1b)} runs for 1B.")
    
    if len(df_1a) > 0 or len(df_1b) > 0:
        print("Generating markdown report...")
        generate_markdown_report(df_1a, df_1b, output_dir)
        print(f"Analysis complete. See '{output_dir}/phase1_report.md'.")
    else:
        print("No Phase 1 data found to analyze.")

if __name__ == "__main__":
    main()
