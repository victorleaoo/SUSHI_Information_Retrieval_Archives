"""
run_new_experiments.py
----------------------
Runs new experiment sets (35 experiments total).

Experiment Groups:
    1-3:   Hybrid ALLFL BM25 experiments with expansion
    4:     Unweighted BM25F TOFS baseline
    5-8:   Learned BM25F single-field runs
    9-10:  ColBERT with expansion variants
    11-12: Unweighted BM25F re-runs (TOF, TFS — corrected from prior mislabeled B runs)
    13-15: ALLFL ColBERT only (T--, TD-, TDN)
    16-18: ColBERT TOFS (T--, TD-, TDN)
    19-20: Unweighted BM25F TOFS (T--, TDN — TD- covered by exp 4)
    21-23: Embeddings TOFS (T--, TD-, TDN)
    24-26: Z (B+C+E) TOFS (T--, TD-, TDN)
    27-29: W (L+C+E) TOFS (T--, TD-, TDN)
    30-32: W TOFS + ALLFL ColBERT hybrid (T--, TD-, TDN)
    33-35: W TOFS + ALLFL ColBERT hybrid + expansion (T--, TD-, TDN)

Run from the /src directory:
    python run_new_experiments.py              # Run ALL experiments
    python run_new_experiments.py 1 3 5        # Run specific experiments
    python run_new_experiments.py 13 14 15     # Run a group
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
TOF_FIELDS   = ['title', 'ocr', 'folderlabel']
TFS_FIELDS   = ['title', 'folderlabel', 'summary']
ALLFL_FIELDS = ['folderlabel']

# Query code → notation segment mapping
QUERY_NOTATION = {'T': 'T--', 'TD': 'TD-', 'TDN': 'TDN'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_output_folder(name: str) -> str:
    path = os.path.abspath(f'../all_runs/{name}')
    os.makedirs(path, exist_ok=True)
    return path


def run_standard(gen, search_fields, query_field, run_folder_name):
    """Run a standard (non-hybrid) experiment with a custom output folder name."""
    print(f"\n{Style.BOLD}{Style.GREEN}> Running: {run_folder_name}{Style.RESET}")
    metrics_folder = make_output_folder(run_folder_name)

    for seed in tqdm(RANDOM_SEED_LIST, desc=f"({run_folder_name})"):
        results = gen.run_single_seed(seed, search_fields, query_field)
        run_name = f'45-Topics-Random-{seed}'
        gen.evaluator.save_run_file(results, RESULTS_PATH, run_name)
        json_path = os.path.join(metrics_folder, f'Random{seed}_TopicsFolderMetrics.json')
        gen.evaluator.evaluate(RESULTS_PATH, json_path)

    print(f"> Generating aggregated metrics in {metrics_folder}...")
    gen.evaluator.generate_aggregated_metrics(metrics_folder, 'random')


def run_hybrid(gen_A, search_field_A, gen_B, search_field_B, query_field, run_folder_name):
    """Run a hybrid (document ranker + ALLFL ranker RRF) experiment."""
    print(f"\n{Style.BOLD}{Style.GREEN}> Running Hybrid: {run_folder_name}{Style.RESET}")
    metrics_folder = make_output_folder(run_folder_name)

    for seed in tqdm(RANDOM_SEED_LIST, desc=f"Hybrid ({run_folder_name})"):
        res_A = gen_A.run_single_seed(seed, search_field_A, query_field)
        res_B = gen_B.run_single_seed(seed, search_field_B, query_field)
        final = perform_hybrid_fusion(res_A, res_B)

        run_name = f'45-Topics-Random-{seed}'
        gen_A.evaluator.save_run_file(final, RESULTS_PATH, run_name)
        json_path = os.path.join(metrics_folder, f'Random{seed}_TopicsFolderMetrics.json')
        gen_A.evaluator.evaluate(RESULTS_PATH, json_path)

    print(f"> Generating aggregated metrics in {metrics_folder}...")
    gen_A.evaluator.generate_aggregated_metrics(metrics_folder, 'random')


def make_allfl_colbert():
    """Shared ALLFL ColBERT RunGenerator for hybrid experiments."""
    gen = RunGenerator(
        searching_fields=[ALLFL_FIELDS],
        query_fields=['TD'],
        models=['colbert'],
        expansion=[],
        all_folders_folder_label=True,
    )
    return gen, ALLFL_FIELDS


def make_allfl_bm25():
    """Shared ALLFL BM25 RunGenerator for hybrid experiments."""
    gen = RunGenerator(
        searching_fields=[ALLFL_FIELDS],
        query_fields=['TD'],
        models=['bm25'],
        expansion=[],
        all_folders_folder_label=True,
    )
    return gen, ALLFL_FIELDS


# ---------------------------------------------------------------------------
# Experiment Definitions
# ---------------------------------------------------------------------------

# ===== FIXED TD- EXPERIMENTS (1-12) =====

def exp1():
    """Exp 1 — U5.TD-.V.TOFS.mx--2.b: V (L+C) TOFS, same_snc+same_box k=2, ALLFL BM25 hybrid."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 1: U5.TD-.V.TOFS.mx--2.b ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TD'],
        models=['bm25', 'colbert'], bm25_tuned=True,
        expansion=['same_snc', 'same_box'], expansion_ceiling_k=2,
        rrf_input='docs',
    )
    gen_B, sf_B = make_allfl_bm25()
    run_hybrid(gen_A, TOFS_FIELDS, gen_B, sf_B, 'TD', 'U5.TD-.V.TOFS.mx--2.b')


def exp2():
    """Exp 2 — U5.TD-.X.TOFS.mx--2.b: X (B+C) TOFS, same_snc+same_box k=2, ALLFL BM25 hybrid."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 2: U5.TD-.X.TOFS.mx--2.b ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TD'],
        models=['bm25', 'colbert'], bm25_tuned=False,
        expansion=['same_snc', 'same_box'], expansion_ceiling_k=2,
        rrf_input='docs',
    )
    gen_B, sf_B = make_allfl_bm25()
    run_hybrid(gen_A, TOFS_FIELDS, gen_B, sf_B, 'TD', 'U5.TD-.X.TOFS.mx--2.b')


def exp3():
    """Exp 3 — U5.TD-.X.TOFS.mx--2.-: X (B+C) TOFS, same_snc+same_box k=2, no hybrid."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 3: U5.TD-.X.TOFS.mx--2.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TD'],
        models=['bm25', 'colbert'], bm25_tuned=False,
        expansion=['same_snc', 'same_box'], expansion_ceiling_k=2,
        rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'TD', 'U5.TD-.X.TOFS.mx--2.-')


def exp4():
    """Exp 4 — U5.TD-.B.TOFS.-.-: Unweighted BM25F TOFS baseline (true B, not L)."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 4: U5.TD-.B.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TD'],
        models=['bm25'], bm25_tuned=False,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'TD', 'U5.TD-.B.TOFS.-.-')


def exp5():
    """Exp 5 — U5.TD-.L.T---.-.-: Learned BM25F, Title only."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 5: U5.TD-.L.T---.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[['title']], query_fields=['TD'],
        models=['bm25'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, ['title'], 'TD', 'U5.TD-.L.T---.-.-')


def exp6():
    """Exp 6 — U5.TD-.L.-O--.-.-: Learned BM25F, OCR only."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 6: U5.TD-.L.-O--.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[['ocr']], query_fields=['TD'],
        models=['bm25'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, ['ocr'], 'TD', 'U5.TD-.L.-O--.-.-')


def exp7():
    """Exp 7 — U5.TD-.L.--F-.-.-: Learned BM25F, FolderLabel only."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 7: U5.TD-.L.--F-.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[['folderlabel']], query_fields=['TD'],
        models=['bm25'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, ['folderlabel'], 'TD', 'U5.TD-.L.--F-.-.-')


def exp8():
    """Exp 8 — U5.TD-.L.---S.-.-: Learned BM25F, Summary only."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 8: U5.TD-.L.---S.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[['summary']], query_fields=['TD'],
        models=['bm25'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, ['summary'], 'TD', 'U5.TD-.L.---S.-.-')


def exp9():
    """Exp 9 — U5.TD-.C.TOFS.m---2.-: ColBERT TOFS, same_snc expansion k=2."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 9: U5.TD-.C.TOFS.m---2.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TD'],
        models=['colbert'],
        expansion=['same_snc'], expansion_ceiling_k=2,
        rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'TD', 'U5.TD-.C.TOFS.m---2.-')


def exp10():
    """Exp 10 — U5.TD-.C.TOFS.--x-2.-: ColBERT TOFS, same_box expansion k=2."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 10: U5.TD-.C.TOFS.--x-2.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TD'],
        models=['colbert'],
        expansion=['same_box'], expansion_ceiling_k=2,
        rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'TD', 'U5.TD-.C.TOFS.--x-2.-')


def exp11():
    """Exp 11 — U5.TD-.B.TOF-.-.-: Unweighted BM25F, TOF fields (corrected B re-run)."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 11: U5.TD-.B.TOF-.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOF_FIELDS], query_fields=['TD'],
        models=['bm25'], bm25_tuned=False,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOF_FIELDS, 'TD', 'U5.TD-.B.TOF-.-.-')


def exp12():
    """Exp 12 — U5.TD-.B.T-FS.-.-: Unweighted BM25F, TFS fields (corrected B re-run)."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 12: U5.TD-.B.T-FS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TFS_FIELDS], query_fields=['TD'],
        models=['bm25'], bm25_tuned=False,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, TFS_FIELDS, 'TD', 'U5.TD-.B.T-FS.-.-')


# ===== QUERY-TYPE SWEEP EXPERIMENTS (13-35) =====
# Each group runs T--, TD-, TDN variants.
# TD- variants that collide with existing runs get _v2 suffix.

def exp13():
    """Exp 13 — U5.T--.-.----.-.c: ALLFL ColBERT only, Title query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 13: U5.T--.-.----.-.c ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[ALLFL_FIELDS], query_fields=['T'],
        models=['colbert'], expansion=[],
        all_folders_folder_label=True,
    )
    run_standard(gen, ALLFL_FIELDS, 'T', 'U5.T--.-.----.-.c')


def exp14():
    """Exp 14 — U5.TD-.-.----.-.c: ALLFL ColBERT only, TD query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 14: U5.TD-.-.----.-.c ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[ALLFL_FIELDS], query_fields=['TD'],
        models=['colbert'], expansion=[],
        all_folders_folder_label=True,
    )
    run_standard(gen, ALLFL_FIELDS, 'TD', 'U5.TD-.-.----.-.c')


def exp15():
    """Exp 15 — U5.TDN.-.----.-.c: ALLFL ColBERT only, TDN query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 15: U5.TDN.-.----.-.c ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[ALLFL_FIELDS], query_fields=['TDN'],
        models=['colbert'], expansion=[],
        all_folders_folder_label=True,
    )
    run_standard(gen, ALLFL_FIELDS, 'TDN', 'U5.TDN.-.----.-.c')


def exp16():
    """Exp 16 — U5.T--.C.TOFS.-.-: ColBERT TOFS, Title query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 16: U5.T--.C.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['T'],
        models=['colbert'], expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'T', 'U5.T--.C.TOFS.-.-')


def exp17():
    """Exp 17 — U5.TD-.C.TOFS.-.-_v2: ColBERT TOFS, TD query (v2 — original exists)."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 17: U5.TD-.C.TOFS.-.-_v2 ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TD'],
        models=['colbert'], expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'TD', 'U5.TD-.C.TOFS.-.-_v2')


def exp18():
    """Exp 18 — U5.TDN.C.TOFS.-.-: ColBERT TOFS, TDN query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 18: U5.TDN.C.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TDN'],
        models=['colbert'], expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'TDN', 'U5.TDN.C.TOFS.-.-')


def exp19():
    """Exp 19 — U5.T--.B.TOFS.-.-: Unweighted BM25F TOFS, Title query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 19: U5.T--.B.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['T'],
        models=['bm25'], bm25_tuned=False,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'T', 'U5.T--.B.TOFS.-.-')


def exp20():
    """Exp 20 — U5.TDN.B.TOFS.-.-: Unweighted BM25F TOFS, TDN query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 20: U5.TDN.B.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TDN'],
        models=['bm25'], bm25_tuned=False,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'TDN', 'U5.TDN.B.TOFS.-.-')

# NOTE: U5.TD-.B.TOFS.-.- is already covered by exp4(). Running exp4 covers this case.


def exp21():
    """Exp 21 — U5.T--.E.TOFS.-.-: Embeddings TOFS, Title query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 21: U5.T--.E.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['T'],
        models=['embeddings'], expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'T', 'U5.T--.E.TOFS.-.-')


def exp22():
    """Exp 22 — U5.TD-.E.TOFS.-.-_v2: Embeddings TOFS, TD query (v2 — original exists)."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 22: U5.TD-.E.TOFS.-.-_v2 ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TD'],
        models=['embeddings'], expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'TD', 'U5.TD-.E.TOFS.-.-_v2')


def exp23():
    """Exp 23 — U5.TDN.E.TOFS.-.-: Embeddings TOFS, TDN query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 23: U5.TDN.E.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TDN'],
        models=['embeddings'], expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'TDN', 'U5.TDN.E.TOFS.-.-')


def exp24():
    """Exp 24 — U5.T--.Z.TOFS.-.-: Z (B+C+E) TOFS, Title query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 24: U5.T--.Z.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['T'],
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=False,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'T', 'U5.T--.Z.TOFS.-.-')


def exp25():
    """Exp 25 — U5.TD-.Z.TOFS.-.-_v2: Z (B+C+E) TOFS, TD query (v2 — original exists)."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 25: U5.TD-.Z.TOFS.-.-_v2 ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TD'],
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=False,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'TD', 'U5.TD-.Z.TOFS.-.-_v2')


def exp26():
    """Exp 26 — U5.TDN.Z.TOFS.-.-: Z (B+C+E) TOFS, TDN query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 26: U5.TDN.Z.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TDN'],
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=False,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'TDN', 'U5.TDN.Z.TOFS.-.-')


def exp27():
    """Exp 27 — U5.T--.W.TOFS.-.-: W (L+C+E) TOFS, Title query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 27: U5.T--.W.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['T'],
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'T', 'U5.T--.W.TOFS.-.-')


def exp28():
    """Exp 28 — U5.TD-.W.TOFS.-.-_v2: W (L+C+E) TOFS, TD query (v2 — original exists)."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 28: U5.TD-.W.TOFS.-.-_v2 ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TD'],
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'TD', 'U5.TD-.W.TOFS.-.-_v2')


def exp29():
    """Exp 29 — U5.TDN.W.TOFS.-.-: W (L+C+E) TOFS, TDN query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 29: U5.TDN.W.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TDN'],
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'TDN', 'U5.TDN.W.TOFS.-.-')


def exp30():
    """Exp 30 — U5.T--.W.TOFS.-.c: W TOFS + ALLFL ColBERT hybrid, Title query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 30: U5.T--.W.TOFS.-.c ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['T'],
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    gen_B, sf_B = make_allfl_colbert()
    run_hybrid(gen_A, TOFS_FIELDS, gen_B, sf_B, 'T', 'U5.T--.W.TOFS.-.c')


def exp31():
    """Exp 31 — U5.TD-.W.TOFS.-.c_v2: W TOFS + ALLFL ColBERT hybrid, TD query (v2)."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 31: U5.TD-.W.TOFS.-.c_v2 ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TD'],
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    gen_B, sf_B = make_allfl_colbert()
    run_hybrid(gen_A, TOFS_FIELDS, gen_B, sf_B, 'TD', 'U5.TD-.W.TOFS.-.c_v2')


def exp32():
    """Exp 32 — U5.TDN.W.TOFS.-.c: W TOFS + ALLFL ColBERT hybrid, TDN query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 32: U5.TDN.W.TOFS.-.c ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TDN'],
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    gen_B, sf_B = make_allfl_colbert()
    run_hybrid(gen_A, TOFS_FIELDS, gen_B, sf_B, 'TDN', 'U5.TDN.W.TOFS.-.c')


def exp33():
    """Exp 33 — U5.T--.W.TOFS.s---2.c: W TOFS + similar_snc k=2 + ALLFL ColBERT, Title query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 33: U5.T--.W.TOFS.s---2.c ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['T'],
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
        expansion=['similar_snc'], expansion_ceiling_k=2,
        rrf_input='docs',
    )
    gen_B, sf_B = make_allfl_colbert()
    run_hybrid(gen_A, TOFS_FIELDS, gen_B, sf_B, 'T', 'U5.T--.W.TOFS.s---2.c')


def exp34():
    """Exp 34 — U5.TD-.W.TOFS.s---2.c: W TOFS + similar_snc k=2 + ALLFL ColBERT, TD query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 34: U5.TD-.W.TOFS.s---2.c ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TD'],
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
        expansion=['similar_snc'], expansion_ceiling_k=2,
        rrf_input='docs',
    )
    gen_B, sf_B = make_allfl_colbert()
    run_hybrid(gen_A, TOFS_FIELDS, gen_B, sf_B, 'TD', 'U5.TD-.W.TOFS.s---2.c')


def exp35():
    """Exp 35 — U5.TDN.W.TOFS.s---2.c: W TOFS + similar_snc k=2 + ALLFL ColBERT, TDN query."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 35: U5.TDN.W.TOFS.s---2.c ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TDN'],
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
        expansion=['similar_snc'], expansion_ceiling_k=2,
        rrf_input='docs',
    )
    gen_B, sf_B = make_allfl_colbert()
    run_hybrid(gen_A, TOFS_FIELDS, gen_B, sf_B, 'TDN', 'U5.TDN.W.TOFS.s---2.c')


# ---------------------------------------------------------------------------
# Registry & Entry Point
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    1: exp1,    2: exp2,    3: exp3,    4: exp4,    5: exp5,
    6: exp6,    7: exp7,    8: exp8,    9: exp9,   10: exp10,
   11: exp11,  12: exp12,  13: exp13,  14: exp14,  15: exp15,
   16: exp16,  17: exp17,  18: exp18,  19: exp19,  20: exp20,
   21: exp21,  22: exp22,  23: exp23,  24: exp24,  25: exp25,
   26: exp26,  27: exp27,  28: exp28,  29: exp29,  30: exp30,
   31: exp31,  32: exp32,  33: exp33,  34: exp34,  35: exp35,
}

if __name__ == "__main__":
    # Parse optional CLI arguments: e.g. "python run_new_experiments.py 1 3 5"
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