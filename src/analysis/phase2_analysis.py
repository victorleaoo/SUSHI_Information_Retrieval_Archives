import os
import json
import pandas as pd

def parse_phase2_results(all_runs_dir):
    data = []
    
    if not os.path.exists(all_runs_dir):
        return pd.DataFrame()

    for folder in os.listdir(all_runs_dir):
        if not folder.startswith("Phase2_"): continue
        
        # Format: Phase2_{run_id}_{aug_method}_{model_name}
        parts = folder.split('_')
        if len(parts) < 4: continue
        
        run_id = parts[1]
        model = parts[-1]
        aug_method = "_".join(parts[2:-1])
        
        stats_path = os.path.join(all_runs_dir, folder, 'all_documents_model_overall_stats.json')
        if not os.path.exists(stats_path):
            continue
            
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            ndcg_mean = stats.get('model_global_ndcg', {}).get('mean', 0.0)
            ndcg_margin = stats.get('model_global_ndcg', {}).get('margin', 0.0)
            
        data.append({
            'RunID': run_id,
            'Strategy': run_id.split('-')[0], # e.g. 2A
            'AugMethod': aug_method,
            'Model': model.upper(),
            'nDCG@5 Mean': ndcg_mean,
            'nDCG@5 CI': ndcg_margin
        })

    return pd.DataFrame(data)

def generate_markdown_report(df_2, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'phase2_report.md')
    
    models_order = ['B', 'E', 'C', 'BC', 'BE', 'CE', 'BCE']
    
    with open(report_path, 'w') as f:
        f.write("# Phase 2: Folder Label Augmentation Analysis\n\n")
        f.write("This report provides a comparison of Phase 2 augmentation strategies over 5 Uniform ECFs.\n\n")
        
        if df_2.empty:
            f.write("*No Phase 2 data found.*\n")
            return
            
        # Group by Strategy
        for strategy in ['2A', '2B', '2C', '2X']:
            df_strat = df_2[df_2['Strategy'] == strategy]
            if df_strat.empty: continue
            
            f.write(f"## Strategy {strategy}\n")
            df_strat = df_strat.sort_values(by='nDCG@5 Mean', ascending=False)
            df_strat['nDCG@5'] = df_strat.apply(lambda row: f"{row['nDCG@5 Mean']:.4f} ± {row['nDCG@5 CI']:.4f}", axis=1)
            f.write(df_strat[['RunID', 'AugMethod', 'Model', 'nDCG@5']].to_markdown(index=False))
            f.write("\n\n")
            
        # Overall Top
        f.write("## 🏆 Overall Top Phase 2 Configurations\n")
        top_n = df_2.sort_values('nDCG@5 Mean', ascending=False).head(15)
        top_n['nDCG@5'] = top_n.apply(lambda row: f"{row['nDCG@5 Mean']:.4f} ± {row['nDCG@5 CI']:.4f}", axis=1)
        top_n.insert(0, 'Rank', range(1, len(top_n) + 1))
        f.write(top_n[['Rank', 'RunID', 'Strategy', 'AugMethod', 'Model', 'nDCG@5']].to_markdown(index=False))
        f.write("\n\n")

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    all_runs_dir = os.path.join(root_dir, 'all_runs')
    output_dir = os.path.join(root_dir, 'results', 'phase2_analysis')
    
    print("Parsing Phase 2 results...")
    df_2 = parse_phase2_results(all_runs_dir)
    
    print(f"Found {len(df_2)} runs for Phase 2.")
    
    if len(df_2) > 0:
        print("Generating markdown report...")
        generate_markdown_report(df_2, output_dir)
        print(f"Analysis complete. See '{output_dir}/phase2_report.md'.")

if __name__ == "__main__":
    main()
