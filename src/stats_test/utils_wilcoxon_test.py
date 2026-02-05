import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class ExperimentAnalyzer:
    def __init__(self, base_runs_path):
        self.base_path = os.path.abspath(base_runs_path)

    def load_run(self, run_folder_name):
        """
        Reads all 'RandomX...' JSON files in a run folder.
        Returns a dict: { seed_filename: { topic_id: ndcg_5_score } }
        """
        run_path = os.path.join(self.base_path, run_folder_name)
        if not os.path.exists(run_path):
            print(f"❌ Error: Folder not found: {run_path}")
            return {}

        seed_data = {}
        files = sorted([f for f in os.listdir(run_path) 
                        if f.startswith("Random") and f.endswith(".json")])
        
        for filename in files:
            try:
                with open(os.path.join(run_path, filename), 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    # Compact extraction
                    seed_data[filename] = {
                        tid: m.get('ndcg_cut_5', 0.0) 
                        for tid, m in content.items() 
                        if isinstance(m, dict)
                    }
            except Exception as e:
                pass

        print(f"-> Loaded {len(seed_data)} seed files from {run_folder_name}")
        return seed_data

    def _align_data(self, data_a, data_b, topic_id=None):
        """
        Internal helper: Aligns two datasets by seed and returns a DataFrame.
        """
        common_seeds = sorted(list(set(data_a.keys()) & set(data_b.keys())))
        
        if not common_seeds:
            return pd.DataFrame()

        rows = []
        for seed in common_seeds:
            if topic_id:
                val_a = data_a[seed].get(topic_id, 0.0)
                val_b = data_b[seed].get(topic_id, 0.0)
            else:
                # Global Mean Calculation
                val_a = np.mean(list(data_a[seed].values())) if data_a[seed] else 0.0
                val_b = np.mean(list(data_b[seed].values())) if data_b[seed] else 0.0
            
            rows.append({'Seed': seed, 'Score_A': val_a, 'Score_B': val_b})
            
        return pd.DataFrame(rows)

    def _plot_differences(self, df, name_a, name_b):
        """Plots the histogram of differences."""
        diffs = df['Diff'].values
        
        plt.figure(figsize=(10, 5))
        n, bins, patches = plt.hist(diffs, bins=15, alpha=0.7, edgecolor='black', linewidth=1.2)
        
        for c, p in zip(bins, patches):
            color = '#2ecc71' if c >= 0 else '#e74c3c' # Green (A wins) vs Red (B wins)
            plt.setp(p, 'facecolor', color)

        plt.axvline(0, color='black', linestyle='--', linewidth=2, label='Tie')
        plt.title(f"Score Differences: {name_a} vs {name_b}")
        plt.xlabel(f"Difference (Positive = {name_a} Better)")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Stats Box
        stats_txt = (f"Mean Diff: {df['Diff'].mean():.4f}\n"
                     f"Max {name_a}: +{df['Diff'].max():.4f}\n"
                     f"Max {name_b}: {df['Diff'].min():.4f}")
        plt.text(0.95, 0.95, stats_txt, transform=plt.gca().transAxes, 
                 fontsize=10, va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        plt.show()

    def _display_rank_table(self, df, name_a, name_b, title="INFLUENTIAL SEEDS"):
        """Calculates and shows the Wilcoxon Signed Rank table."""
        # Calculate ranks on absolute differences
        df['Abs_Diff'] = df['Diff'].abs()
        
        # Wilcoxon logic: Ranks are based on Abs_Diff
        df['Rank'] = stats.rankdata(df['Abs_Diff'], method='average')
        
        # Signed Rank: Rank * Sign of Diff
        df['Signed_Rank'] = df['Rank'] * np.sign(df['Diff'])
        
        # Sort by impact (Absolute Rank) descending
        display_df = df.sort_values(by='Rank', ascending=False).reset_index(drop=True)
        
        print(f"\n📊 --- {title} (Top 10) ---")
        print("Rows are sorted by 'Rank' (Magnitude of difference).")
        print(f"Green = {name_a} won | Red = {name_b} won")

        # Styling
        def highlight(val):
            # Colors: Greenish for positive, Reddish for negative
            color = '#d4edda' if val > 0 else '#f8d7da' if val < 0 else 'white'
            return f'background-color: {color}'

        cols_to_show = ['Seed', 'Score_A', 'Score_B', 'Diff', 'Rank', 'Signed_Rank']
        
        # Display top 15 rows
        styled = display_df[cols_to_show].head(15).style.applymap(highlight, subset=['Diff', 'Signed_Rank'])\
                  .format({'Score_A': "{:.4f}", 'Score_B': "{:.4f}", 'Diff': "{:+.4f}", 'Rank': "{:.1f}"})
        display(styled)

    def compare(self, data_a, data_b, name_a="Run A", name_b="Run B", topic_id=None, show_table=True):
        """
        Main comparison method. 
        Args:
            topic_id: If None, compares Global Averages. If set, compares that topic.
            show_table: If True, displays the Signed Rank Table.
        """
        # 1. Align Data
        df = self._align_data(data_a, data_b, topic_id)
        
        if df.empty:
            print("❌ No matching seeds found.")
            return

        # 2. Compute Differences
        df['Diff'] = df['Score_A'] - df['Score_B']
        
        # 3. Basic Stats
        wins_a = (df['Diff'] > 0).sum()
        wins_b = (df['Diff'] < 0).sum()
        mean_a, mean_b = df['Score_A'].mean(), df['Score_B'].mean()
        
        label = f"Topic {topic_id}" if topic_id else "Global Average (All Topics)"
        print(f"\n🧪 --- ANALYSIS: {label} ---")
        print(f"Sample Size: {len(df)}")
        print(f"Mean {name_a}: {mean_a:.4f} | Wins: {wins_a}")
        print(f"Mean {name_b}: {mean_b:.4f} | Wins: {wins_b}")

        # 4. Wilcoxon Test
        try:
            stat, p_val = stats.wilcoxon(df['Score_A'], df['Score_B'])
            print(f"Wilcoxon p-value: {p_val:.5f}")
            if p_val < 0.05:
                winner = name_a if mean_a > mean_b else name_b
                print(f"✅ SIGNIFICANT: {winner} is better.")
            else:
                print(f"❌ NOT SIGNIFICANT.")
        except ValueError:
            print("⚠️ Identical scores, cannot run Wilcoxon.")

        # 5. Visuals
        self._plot_differences(df, name_a, name_b)
        
        # 6. Rank Table (Now works for Global too!)
        if show_table:
            table_title = f"RANK BREAKDOWN ({label})"
            self._display_rank_table(df, name_a, name_b, title=table_title)

    def scan_significant_topics(self, data_a, data_b, name_a="Run A", name_b="Run B"):
        """Iterates all topics to find significant ones."""
        first_seed = list(data_a.keys())[0]
        all_topics = sorted(list(data_a[first_seed].keys()))
        
        results = []
        print(f"🔎 Scanning {len(all_topics)} topics...")
        
        for tid in all_topics:
            df = self._align_data(data_a, data_b, tid)
            if df.empty: continue
            
            try:
                s, p = stats.wilcoxon(df['Score_A'], df['Score_B'])
                if p < 0.05:
                    mean_a, mean_b = df['Score_A'].mean(), df['Score_B'].mean()
                    winner = name_a if mean_a > mean_b else name_b
                    results.append({
                        'Topic': tid, 'Winner': winner, 'p-value': p, 
                        'Diff': mean_a - mean_b
                    })
            except: pass

        if not results:
            print("❌ No significant topics found.")
            return

        res_df = pd.DataFrame(results).sort_values('p-value').reset_index(drop=True)
        
        # Color the winner column
        def color_win(val):
            return f'background-color: {"#d4edda" if val == name_a else "#f8d7da"}'
            
        display(res_df.style.applymap(color_win, subset=['Winner'])
                .format({'p-value': "{:.5f}", 'Diff': "{:+.4f}"}))