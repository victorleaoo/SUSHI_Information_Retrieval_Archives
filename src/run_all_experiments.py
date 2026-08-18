"""
run_all_experiments.py
----------------------
Runs all planned experiment sets in sequence:

    1. All Documents TOFS       - BM25F, ColBERT, Embeddings
    2. Skewed TFS               - BM25F, ColBERT, Embeddings
    3. Skewed TOFS              - BM25F, ColBERT, Embeddings + ALLFL ColBERT RRF (no expansion)
    4. Skewed TOFS + SMS-2 exp  - BM25F, ColBERT, Embeddings (expansion_ceiling_k=2)
    5. Skewed TOFS              - BM25F, ColBERT, Embeddings + ALLFL ColBERT RRF (no expansion, variant)
    6. Skewed TOFS + SMS-2 exp  - BM25F, ColBERT, Embeddings + ALLFL ColBERT RRF

Run from the /src directory:
    python run_all_experiments.py

To run only a subset, pass experiment numbers as arguments:
    python run_all_experiments.py 1 3 5
"""

import os
import sys
from tqdm import tqdm

from run_generator import RunGenerator, RANDOM_SEED_LIST, RESULTS_PATH, Style
from hybrid_models import perform_hybrid_fusion

# ---------------------------------------------------------------------------
# Field sets
# ---------------------------------------------------------------------------

TOFS_FIELDS  = ['title', 'ocr', 'folderlabel', 'summary']
TFS_FIELDS   = ['title', 'folderlabel', 'summary']
ALLFL_FIELDS = ['folderlabel']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_output_folder(name: str) -> str:
    path = os.path.abspath(f'../all_runs/{name}')
    os.makedirs(path, exist_ok=True)
    return path


def make_allfl_colbert():
    """Shared ALLFL ColBERT RunGenerator (Config B) used in all hybrid experiments."""
    gen_B = RunGenerator(
        searching_fields=[ALLFL_FIELDS],
        query_fields=['TD'],
        models=['colbert'],
        expansion=[],
        all_folders_folder_label=True,
    )
    return gen_B, ALLFL_FIELDS


def run_hybrid(gen_A, search_field_A, gen_B, search_field_B, run_folder_name):
    """Shared loop for hybrid (document model + ALLFL ColBERT RRF) experiments."""
    metrics_folder = make_output_folder(run_folder_name)

    for seed in tqdm(RANDOM_SEED_LIST, desc=f"Hybrid ({run_folder_name})"):
        res_A = gen_A.run_single_seed(seed, search_field_A, 'TD')
        res_B = gen_B.run_single_seed(seed, search_field_B, 'TD')
        final = perform_hybrid_fusion(res_A, res_B)

        run_name = f'45-Topics-Random-{seed}'
        gen_A.evaluator.save_run_file(final, RESULTS_PATH, run_name)

        json_path = os.path.join(metrics_folder, f'Random{seed}_TopicsFolderMetrics.json')
        gen_A.evaluator.evaluate(RESULTS_PATH, json_path)

    print(f"> Generating aggregated metrics in {metrics_folder}...")
    gen_A.evaluator.generate_aggregated_metrics(metrics_folder, 'random')


# ---------------------------------------------------------------------------
# Experiment Definitions
# ---------------------------------------------------------------------------

def exp1_all_docs_tofs():
    """Exp 1 - All Documents TOFS: BM25F + ColBERT + Embeddings (oracle)."""
    print(f"\n{Style.BOLD}{Style.GREEN}=== EXP 1: All Documents TOFS (BM25F + ColBERT + Embeddings) ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS],
        query_fields=['TD'],
        run_type='all_documents',
        models=['bm25', 'embeddings', 'colbert'],
        sampling='uniform',
        expansion=[],
        rrf_input='docs',
    )
    gen.run_experiments()


def exp2_skewed_tfs():
    """Exp 2 - Skewed TFS (Title + FolderLabel + Summary): BM25F + ColBERT + Embeddings."""
    print(f"\n{Style.BOLD}{Style.GREEN}=== EXP 2: Skewed TFS (BM25F + ColBERT + Embeddings) ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TFS_FIELDS],
        query_fields=['TD'],
        run_type='random',
        models=['bm25', 'embeddings', 'colbert'],
        sampling='uneven',
        expansion=[],
        rrf_input='docs',
    )
    gen.run_experiments()


def exp3_skewed_tofs_hybrid_no_expansion():
    """Exp 3 - Skewed TOFS: BM25F + ColBERT + Embeddings + ALLFL ColBERT RRF (no expansion)."""
    print(f"\n{Style.BOLD}{Style.GREEN}=== EXP 3: Skewed TOFS + ALLFL ColBERT RRF (no expansion) ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS],
        query_fields=['TD'],
        run_type='random',
        models=['bm25', 'embeddings', 'colbert'],
        sampling='uneven',
        expansion=[],
        rrf_input='docs',
    )
    gen_B, search_field_B = make_allfl_colbert()
    run_hybrid(
        gen_A, TOFS_FIELDS,
        gen_B, search_field_B,
        run_folder_name="HYBRID-SKEWED-TOFS-NEX-ALLFL-COLBERT_NE_TD_BM25-EMBEDDINGS-COLBERT",
    )


def exp4_skewed_tofs_similar_snc_k2():
    """Exp 4 - Skewed TOFS: BM25F + ColBERT + Embeddings + similar_snc expansion (ceiling_k=2)."""
    print(f"\n{Style.BOLD}{Style.GREEN}=== EXP 4: Skewed TOFS + similar_snc expansion (k=2) ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS],
        query_fields=['TD'],
        run_type='random',
        models=['bm25', 'embeddings', 'colbert'],
        sampling='uneven',
        expansion=['similar_snc'],
        expansion_ceiling_k=2,
        rrf_input='docs',
    )
    gen.run_experiments()


def exp5_skewed_tofs_hybrid_no_expansion_v2():
    """Exp 5 - Skewed TOFS: BM25F + ColBERT + Embeddings + ALLFL ColBERT RRF (no expansion, variant)."""
    print(f"\n{Style.BOLD}{Style.GREEN}=== EXP 5: Skewed TOFS + ALLFL ColBERT RRF (no expansion, variant) ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS],
        query_fields=['TD'],
        run_type='random',
        models=['bm25', 'embeddings', 'colbert'],
        sampling='uneven',
        expansion=[],
        rrf_input='docs',
    )
    gen_B, search_field_B = make_allfl_colbert()
    run_hybrid(
        gen_A, TOFS_FIELDS,
        gen_B, search_field_B,
        run_folder_name="HYBRID-SKEWED-TOFS-NEX-ALLFL-COLBERT_NE_TD_BM25-EMBEDDINGS-COLBERT-V2",
    )


def exp6_skewed_tofs_similar_snc_k2_hybrid():
    """Exp 6 - Skewed TOFS + similar_snc (k=2) + ALLFL ColBERT RRF."""
    print(f"\n{Style.BOLD}{Style.GREEN}=== EXP 6: Skewed TOFS + similar_snc (k=2) + ALLFL ColBERT RRF ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS],
        query_fields=['TD'],
        run_type='random',
        models=['bm25', 'embeddings', 'colbert'],
        sampling='uneven',
        expansion=['similar_snc'],
        expansion_ceiling_k=2,
        rrf_input='docs',
    )
    gen_B, search_field_B = make_allfl_colbert()
    run_hybrid(
        gen_A, TOFS_FIELDS,
        gen_B, search_field_B,
        run_folder_name="HYBRID-SKEWED-TOFS-SMS-2-ALLFL-COLBERT_SMS-2_TD_BM25-EMBEDDINGS-COLBERT",
    )


# ---------------------------------------------------------------------------
# Registry & Entry Point
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    1: exp1_all_docs_tofs,
    2: exp2_skewed_tfs,
    3: exp3_skewed_tofs_hybrid_no_expansion,
    4: exp4_skewed_tofs_similar_snc_k2,
    5: exp5_skewed_tofs_hybrid_no_expansion_v2,
    6: exp6_skewed_tofs_similar_snc_k2_hybrid,
}

if __name__ == "__main__":
    # Parse optional CLI arguments: e.g. "python run_all_experiments.py 1 3 5"
    if len(sys.argv) > 1:
        selected = [int(a) for a in sys.argv[1:] if a.isdigit()]
    else:
        selected = list(EXPERIMENTS.keys())

    print(f"{Style.BOLD}{Style.CYAN}Running experiments: {selected}{Style.RESET}")

    for exp_id in selected:
        if exp_id not in EXPERIMENTS:
            print(f"{Style.WARNING}Warning: experiment {exp_id} not found, skipping.{Style.RESET}")
            continue
        try:
            EXPERIMENTS[exp_id]()
            print(f"{Style.GREEN}> Experiment {exp_id} complete.{Style.RESET}\n")
        except Exception as e:
            print(f"{Style.FAIL}> Experiment {exp_id} failed: {e}{Style.RESET}\n")
            raise

    print(f"\n{Style.BOLD}{Style.GREEN}All done!{Style.RESET}")
