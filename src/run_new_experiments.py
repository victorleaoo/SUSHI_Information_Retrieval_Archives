"""
run_new_experiments.py
----------------------
Runs the Title-only (T--) query sweep (15 experiments total).

Each experiment mirrors an existing TD- (Title+Description) configuration but
replaces the query field with T (Title only).

Experiment Groups:
    1:     U5.T--.V.TOFS.mx--2.b  - V (L+C) TOFS, same_snc+same_box k=2, ALLFL BM25 hybrid
    2:     U5.T--.X.TOFS.mx--2.b  - X (B+C) TOFS, same_snc+same_box k=2, ALLFL BM25 hybrid
    3:     U5.T--.L.TOFS.mx--2.-  - L TOFS, same_snc+same_box k=2, no hybrid
    4:     U5.T--.B.TOFS.-.-      - Unweighted BM25F TOFS baseline
    5-8:   U5.T--.L single-field  - Learned BM25F, one field at a time (O, S, T, F)
    9-13:  U1-U5.T--.W.TOFS.-.-   - Learned BM25F+ColBERT+Embeddings RRF, docs/box sweep
    14:    K5.T--.W.TOFS.-.-      - W TOFS, skewed sampling
    15:    A-.T--.W.TOFS.-.-      - W TOFS, all-documents (oracle)
    16:    U5.T--.V.TOFS.mx--2.b (Official ECF) - Official NTCIR-18 protocol counterpart of exp1
    17:    U5.T--.X.TOFS.mx--2.b (Official ECF) - Official NTCIR-18 protocol counterpart of exp2
    18:    U5.T--.L.TOFS.mx--2.- (Official ECF) - Official NTCIR-18 protocol counterpart of exp3
    19-22: U5.T--.L single-field (Official ECF) - Official NTCIR-18 protocol counterparts of exp5-8

Run from the /src directory:
    python run_new_experiments.py              # Run ALL experiments
    python run_new_experiments.py 1 3 5        # Run specific experiments
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
ALLFL_FIELDS = ['folderlabel']

QUERY_FIELD = 'TD'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_output_folder(name: str) -> str:
    path = os.path.abspath(f'../all_runs/{name}')
    os.makedirs(path, exist_ok=True)
    return path


def run_standard(gen, search_fields, query_field, run_folder_name):
    """Run a standard random experiment (30 seeds) with a custom output folder name."""
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


def run_standard_official_ecf(gen, search_fields, query_field, run_folder_name):
    """Run a standard (non-hybrid) experiment under the NTCIR-18 official ECF protocol (single pass, no seed loop)."""
    print(f"\n{Style.BOLD}{Style.GREEN}> Running (Official ECF): {run_folder_name}{Style.RESET}")
    metrics_folder = make_output_folder(run_folder_name)

    results = gen.run_official_ecf(search_fields, query_field)
    run_name = 'OfficialECF-3Sets-45Topics'
    gen.evaluator.save_run_file(results, RESULTS_PATH, run_name)
    json_path = os.path.join(metrics_folder, 'OfficialECF_TopicsFolderMetrics.json')
    gen.evaluator.evaluate(RESULTS_PATH, json_path)
    gen.evaluator.generate_aggregated_metrics(metrics_folder, 'official_ecf')


def run_hybrid_official_ecf(gen_A, search_field_A, gen_B, search_field_B, query_field, run_folder_name):
    """Run a hybrid (document ranker + ALLFL ranker RRF) experiment under the NTCIR-18 official ECF protocol."""
    print(f"\n{Style.BOLD}{Style.GREEN}> Running Hybrid (Official ECF): {run_folder_name}{Style.RESET}")
    metrics_folder = make_output_folder(run_folder_name)

    res_A = gen_A.run_official_ecf(search_field_A, query_field)
    res_B = gen_B.run_official_ecf(search_field_B, query_field)
    final = perform_hybrid_fusion(res_A, res_B)

    run_name = 'OfficialECF-3Sets-45Topics'
    gen_A.evaluator.save_run_file(final, RESULTS_PATH, run_name)
    json_path = os.path.join(metrics_folder, 'OfficialECF_TopicsFolderMetrics.json')
    gen_A.evaluator.evaluate(RESULTS_PATH, json_path)
    gen_A.evaluator.generate_aggregated_metrics(metrics_folder, 'official_ecf')


def run_all_documents(gen, search_fields, query_field, run_folder_name):
    """Run a single all-documents (oracle) experiment with a custom output folder name."""
    print(f"\n{Style.BOLD}{Style.CYAN}> Running All Documents: {run_folder_name}{Style.RESET}")
    metrics_folder = make_output_folder(run_folder_name)

    results = gen.run_single_seed(0, search_fields, query_field)
    run_name = '45-Topics-AllDocuments'
    gen.evaluator.save_run_file(results, RESULTS_PATH, run_name)
    json_path = os.path.join(metrics_folder, 'AllDocuments_TopicsFolderMetrics.json')
    gen.evaluator.evaluate(RESULTS_PATH, json_path)
    gen.evaluator.generate_aggregated_metrics(metrics_folder, 'all_documents')


def make_allfl_bm25():
    """Shared ALLFL BM25 RunGenerator for hybrid experiments."""
    gen = RunGenerator(
        searching_fields=[ALLFL_FIELDS],
        query_fields=[QUERY_FIELD],
        models=['bm25'],
        bm25_tuned=False,
        expansion=[],
        all_folders_folder_label=True,
    )
    return gen, ALLFL_FIELDS


# ---------------------------------------------------------------------------
# Experiment Definitions
# ---------------------------------------------------------------------------

def exp1():
    """Exp 1 — U5.T--.V.TOFS.mx--2.b: V (L+C) TOFS, same_snc+same_box k=2, ALLFL BM25 hybrid."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 1: U5.T--.V.TOFS.mx--2.b ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=[QUERY_FIELD],
        models=['bm25', 'colbert'], bm25_tuned=True,
        expansion=['same_snc', 'same_box'], expansion_ceiling_k=2,
        rrf_input='docs',
    )
    gen_B, sf_B = make_allfl_bm25()
    run_hybrid(gen_A, TOFS_FIELDS, gen_B, sf_B, QUERY_FIELD, 'U5.T--.V.TOFS.mx--2.b')


def exp2():
    """Exp 2 — U5.T--.X.TOFS.mx--2.b: X (B+C) TOFS, same_snc+same_box k=2, ALLFL BM25 hybrid."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 2: U5.T--.X.TOFS.mx--2.b ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=[QUERY_FIELD],
        models=['bm25', 'colbert'], bm25_tuned=False,
        expansion=['same_snc', 'same_box'], expansion_ceiling_k=2,
        rrf_input='docs',
    )
    gen_B, sf_B = make_allfl_bm25()
    run_hybrid(gen_A, TOFS_FIELDS, gen_B, sf_B, QUERY_FIELD, 'U5.T--.X.TOFS.mx--2.b')


def exp1_official():
    """Exp 1 (Official ECF) — U5.T--.V.TOFS.mx--2.b under the NTCIR-18 official protocol."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 1 (Official ECF): U5.T--.V.TOFS.mx--2.b ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=[QUERY_FIELD],
        run_type='official_ecf',
        models=['bm25', 'colbert'], bm25_tuned=True,
        expansion=['same_snc', 'same_box'], expansion_ceiling_k=2,
        rrf_input='docs',
    )
    gen_B, sf_B = make_allfl_bm25()
    gen_B.run_type = 'official_ecf'
    run_hybrid_official_ecf(gen_A, TOFS_FIELDS, gen_B, sf_B, QUERY_FIELD, 'U5.T--.V.TOFS.mx--2.b.OfficialECF')


def exp2_official():
    """Exp 2 (Official ECF) — U5.T--.X.TOFS.mx--2.b under the NTCIR-18 official protocol."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 2 (Official ECF): U5.T--.X.TOFS.mx--2.b ==={Style.RESET}")
    gen_A = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=[QUERY_FIELD],
        run_type='official_ecf',
        models=['bm25', 'colbert'], bm25_tuned=False,
        expansion=['same_snc', 'same_box'], expansion_ceiling_k=2,
        rrf_input='docs',
    )
    gen_B, sf_B = make_allfl_bm25()
    gen_B.run_type = 'official_ecf'
    run_hybrid_official_ecf(gen_A, TOFS_FIELDS, gen_B, sf_B, QUERY_FIELD, 'U5.T--.X.TOFS.mx--2.b.OfficialECF')


def exp3():
    """Exp 3 — U5.T--.L.TOFS.mx--2.-: L TOFS, same_snc+same_box k=2, no hybrid."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 3: U5.T--.L.TOFS.mx--2.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=[QUERY_FIELD],
        models=['bm25'], bm25_tuned=True,
        expansion=['same_snc', 'same_box'], expansion_ceiling_k=2,
        rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, QUERY_FIELD, 'U5.T--.L.TOFS.mx--2.-')


def exp3_official():
    """Exp 3 (Official ECF) — U5.T--.L.TOFS.mx--2.- under the NTCIR-18 official protocol."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 3 (Official ECF): U5.T--.L.TOFS.mx--2.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=[QUERY_FIELD],
        run_type='official_ecf',
        models=['bm25'], bm25_tuned=True,
        expansion=['same_snc', 'same_box'], expansion_ceiling_k=2,
        rrf_input='docs',
    )
    run_standard_official_ecf(gen, TOFS_FIELDS, QUERY_FIELD, 'U5.T--.L.TOFS.mx--2.-.OfficialECF')


def exp4():
    """Exp 4 — U5.T--.B.TOFS.-.-: Unweighted BM25F TOFS baseline."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 4: U5.T--.B.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=[QUERY_FIELD],
        models=['bm25'], bm25_tuned=False,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, QUERY_FIELD, 'U5.T--.B.TOFS.-.-')


def exp_L_TOFS_NEX():
    """Exp — U5.T--.L.TOFS.-.-: Learned/tuned BM25F TOFS baseline, no expansion (L counterpart of exp4)."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP: U5.T--.L.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['T'],
        models=['bm25'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, 'T', 'U5.T--.L.TOFS.-.-')


def exp4_official():
    """Exp 4 (Official ECF) — U5.T--.B.TOFS.-.-: Unweighted BM25F TOFS baseline, under the NTCIR-18 official protocol."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 4 (Official ECF): U5.T--.B.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=[QUERY_FIELD],
        run_type='official_ecf',
        models=['bm25'], bm25_tuned=False,
        expansion=[], rrf_input='docs',
    )
    run_standard_official_ecf(gen, TOFS_FIELDS, QUERY_FIELD, 'U5.T--.B.TOFS.-.-.OfficialECF')


def exp5():
    """Exp 5 — U5.T--.L.-O--.-.-: Learned BM25F, OCR only."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 5: U5.T--.L.-O--.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[['ocr']], query_fields=[QUERY_FIELD],
        models=['bm25'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, ['ocr'], QUERY_FIELD, 'U5.T--.L.-O--.-.-')


def exp6():
    """Exp 6 — U5.T--.L.---S.-.-: Learned BM25F, Summary only."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 6: U5.T--.L.---S.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[['summary']], query_fields=[QUERY_FIELD],
        models=['bm25'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, ['summary'], QUERY_FIELD, 'U5.T--.L.---S.-.-')


def exp7():
    """Exp 7 — U5.T--.L.T---.-.-: Learned BM25F, Title only."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 7: U5.T--.L.T---.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[['title']], query_fields=[QUERY_FIELD],
        models=['bm25'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, ['title'], QUERY_FIELD, 'U5.T--.L.T---.-.-')


def exp8():
    """Exp 8 — U5.T--.L.--F-.-.-: Learned BM25F, FolderLabel only."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 8: U5.T--.L.--F-.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[['folderlabel']], query_fields=[QUERY_FIELD],
        models=['bm25'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, ['folderlabel'], QUERY_FIELD, 'U5.T--.L.--F-.-.-')


def exp5_official():
    """Exp 5 (Official ECF) — U5.T--.L.-O--.-.-: Learned BM25F, OCR only, under the NTCIR-18 official protocol."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 5 (Official ECF): U5.T--.L.-O--.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[['ocr']], query_fields=[QUERY_FIELD],
        run_type='official_ecf',
        models=['bm25'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard_official_ecf(gen, ['ocr'], QUERY_FIELD, 'U5.T--.L.-O--.-.-.OfficialECF')


def exp6_official():
    """Exp 6 (Official ECF) — U5.T--.L.---S.-.-: Learned BM25F, Summary only, under the NTCIR-18 official protocol."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 6 (Official ECF): U5.T--.L.---S.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[['summary']], query_fields=[QUERY_FIELD],
        run_type='official_ecf',
        models=['bm25'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard_official_ecf(gen, ['summary'], QUERY_FIELD, 'U5.T--.L.---S.-.-.OfficialECF')


def exp7_official():
    """Exp 7 (Official ECF) — U5.T--.L.T---.-.-: Learned BM25F, Title only, under the NTCIR-18 official protocol."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 7 (Official ECF): U5.T--.L.T---.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[['title']], query_fields=[QUERY_FIELD],
        run_type='official_ecf',
        models=['bm25'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard_official_ecf(gen, ['title'], QUERY_FIELD, 'U5.T--.L.T---.-.-.OfficialECF')


def exp8_official():
    """Exp 8 (Official ECF) — U5.T--.L.--F-.-.-: Learned BM25F, FolderLabel only, under the NTCIR-18 official protocol."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 8 (Official ECF): U5.T--.L.--F-.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[['folderlabel']], query_fields=[QUERY_FIELD],
        run_type='official_ecf',
        models=['bm25'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_standard_official_ecf(gen, ['folderlabel'], QUERY_FIELD, 'U5.T--.L.--F-.-.-.OfficialECF')


def _exp_w_tofs_docs_per_box(docs_per_box, run_folder_name):
    """Shared body for the U1-U5 W (L+C+E) TOFS docs/box sweep."""
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=[QUERY_FIELD],
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
        docs_per_box=docs_per_box,
    )
    run_standard(gen, TOFS_FIELDS, QUERY_FIELD, run_folder_name)


def exp9():
    """Exp 9 — U1.T--.W.TOFS.-.-: W TOFS, 1 doc/box."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 9: U1.T--.W.TOFS.-.- ==={Style.RESET}")
    _exp_w_tofs_docs_per_box(1, 'U1.T--.W.TOFS.-.-')


def exp10():
    """Exp 10 — U2.T--.W.TOFS.-.-: W TOFS, 2 docs/box."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 10: U2.T--.W.TOFS.-.- ==={Style.RESET}")
    _exp_w_tofs_docs_per_box(2, 'U2.T--.W.TOFS.-.-')


def exp11():
    """Exp 11 — U3.T--.W.TOFS.-.-: W TOFS, 3 docs/box."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 11: U3.T--.W.TOFS.-.- ==={Style.RESET}")
    _exp_w_tofs_docs_per_box(3, 'U3.T--.W.TOFS.-.-')


def exp12():
    """Exp 12 — U4.T--.W.TOFS.-.-: W TOFS, 4 docs/box."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 12: U4.T--.W.TOFS.-.- ==={Style.RESET}")
    _exp_w_tofs_docs_per_box(4, 'U4.T--.W.TOFS.-.-')


def exp13():
    """Exp 13 — U5.T--.W.TOFS.-.-: W TOFS, 5 docs/box."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 13: U5.T--.W.TOFS.-.- ==={Style.RESET}")
    _exp_w_tofs_docs_per_box(5, 'U5.T--.W.TOFS.-.-')


def exp14():
    """Exp 14 — K5.T--.W.TOFS.-.-: W TOFS, skewed sampling."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 14: K5.T--.W.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=[QUERY_FIELD],
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
        sampling='uneven',
        expansion=[], rrf_input='docs',
    )
    run_standard(gen, TOFS_FIELDS, QUERY_FIELD, 'K5.T--.W.TOFS.-.-')


def exp15():
    """Exp 15 — A-.T--.W.TOFS.-.-: W TOFS, all documents (oracle)."""
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP 15: A-.T--.W.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=[QUERY_FIELD],
        run_type='all_documents',
        models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
        expansion=[], rrf_input='docs',
    )
    run_all_documents(gen, TOFS_FIELDS, QUERY_FIELD, 'A-.T--.W.TOFS.-.-')


# ---------------------------------------------------------------------------
# Registry & Entry Point
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    # 1: exp1,   2: exp2,
    # 3: exp3,
    # 4: exp4, 5: exp5,   6: exp6,   7: exp7,   8: exp8,
    # 9: exp9,   10: exp10, 11: exp11, 12: exp12, 13: exp13,
    # 14: exp14, 15: exp15,
    16: exp1_official,  # Official ECF counterpart of exp1 (U5.T--.V.TOFS.mx--2.b)
    17: exp2_official,  # Official ECF counterpart of exp2 (U5.T--.X.TOFS.mx--2.b)
    18: exp3_official,  # Official ECF counterpart of exp3 (U5.T--.L.TOFS.mx--2.-)
    19: exp5_official,  # Official ECF counterpart of exp5 (U5.T--.L.-O--.-.-)
    20: exp6_official,  # Official ECF counterpart of exp6 (U5.T--.L.---S.-.-)
    21: exp7_official,  # Official ECF counterpart of exp7 (U5.T--.L.T---.-.-)
    22: exp8_official,  # Official ECF counterpart of exp8 (U5.T--.L.--F-.-.-)
    23: exp4_official,  # Official ECF counterpart of exp4 (U5.T--.B.TOFS.-.-)
    24: exp_L_TOFS_NEX,  # U5.T--.L.TOFS.-.-: L (tuned) counterpart of exp4, no expansion
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
