"""
run_new_experiments.py
----------------------
Full re-run of the 8 core configurations for EVERY query field (T, TD, TDN).

8 configurations x 3 query fields = 24 experiments, all under the 'random'
protocol (30 seeds from RANDOM_SEED_LIST).

Configurations (shown with the T-- query tag; TD- / TDN are the counterparts):
    U5.<QF>.V.TOFS.mx--2.b  - Tuned BM25F + ColBERT, TOFS, same_snc+same_box k=2,
                              hybrid RRF with the ALLFL BM25 ranker
    U5.<QF>.X.TOFS.mx--2.b  - Untuned BM25F + ColBERT, TOFS, same_snc+same_box k=2,
                              hybrid RRF with the ALLFL BM25 ranker
    U5.<QF>.L.TOFS.mx--2.-  - Tuned BM25F, TOFS, same_snc+same_box k=2, no hybrid
    U5.<QF>.B.TOFS.-.-      - Untuned BM25F, TOFS, no expansion
    U5.<QF>.L.-O--.-.-      - Tuned BM25F, OCR only
    U5.<QF>.L.---S.-.-      - Tuned BM25F, Summary only
    U5.<QF>.L.T---.-.-      - Tuned BM25F, Title only
    U5.<QF>.L.--F-.-.-      - Tuned BM25F, FolderLabel only

Experiment ids:
    T   ->  1..8      (folders U5.T--.*)
    TD  -> 11..18     (folders U5.TD-.*)
    TDN -> 21..28     (folders U5.TDN.*)

Run from the /src directory:
    python run_new_experiments.py              # Run ALL 24 experiments
    python run_new_experiments.py 1 3 5        # Run specific experiment ids
    python run_new_experiments.py T TDN        # Run every config for those query fields
"""

import os
import sys
from tqdm import tqdm

from run_generator import RunGenerator, RANDOM_SEED_LIST, RESULTS_PATH, Style
from hybrid_models import perform_hybrid_fusion

# ---------------------------------------------------------------------------
# Field sets & query fields
# ---------------------------------------------------------------------------

TOFS_FIELDS  = ['title', 'ocr', 'folderlabel', 'summary']
ALLFL_FIELDS = ['folderlabel']

# Query field -> tag used in the run folder name (2nd slot, always 3 chars).
QUERY_FIELD_TAGS = {'T': 'T--', 'TD': 'TD-', 'TDN': 'TDN'}

# Base id offset per query field: T -> 1..8, TD -> 11..18, TDN -> 21..28
QUERY_FIELD_ID_BASE = {'T': 0, 'TD': 10, 'TDN': 20}


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


def make_allfl_bm25(query_field):
    """Shared ALLFL BM25 RunGenerator used as ranker B in the hybrid experiments."""
    gen = RunGenerator(
        searching_fields=[ALLFL_FIELDS],
        query_fields=[query_field],
        models=['bm25'],
        bm25_tuned=False,
        expansion=[],
        all_folders_folder_label=True,
    )
    return gen, ALLFL_FIELDS


# ---------------------------------------------------------------------------
# Configuration table (query-field agnostic)
# ---------------------------------------------------------------------------
# Each entry:
#   offset      : id offset within the query-field block (1..8)
#   suffix      : folder-name suffix appended to 'U5.<QF_TAG>.'
#   fields      : searching fields
#   hybrid      : whether ranker B (ALLFL BM25) is fused in
#   kwargs      : RunGenerator kwargs for ranker A
CONFIGS = [
    {
        'offset': 1,
        'suffix': 'V.TOFS.mx--2.b',
        'fields': TOFS_FIELDS,
        'hybrid': True,
        'kwargs': dict(models=['bm25', 'colbert'], bm25_tuned=True,
                       expansion=['same_snc', 'same_box'], expansion_ceiling_k=2,
                       rrf_input='docs'),
    },
    {
        'offset': 2,
        'suffix': 'X.TOFS.mx--2.b',
        'fields': TOFS_FIELDS,
        'hybrid': True,
        'kwargs': dict(models=['bm25', 'colbert'], bm25_tuned=False,
                       expansion=['same_snc', 'same_box'], expansion_ceiling_k=2,
                       rrf_input='docs'),
    },
    {
        'offset': 3,
        'suffix': 'L.TOFS.mx--2.-',
        'fields': TOFS_FIELDS,
        'hybrid': False,
        'kwargs': dict(models=['bm25'], bm25_tuned=True,
                       expansion=['same_snc', 'same_box'], expansion_ceiling_k=2,
                       rrf_input='docs'),
    },
    {
        'offset': 4,
        'suffix': 'B.TOFS.-.-',
        'fields': TOFS_FIELDS,
        'hybrid': False,
        'kwargs': dict(models=['bm25'], bm25_tuned=False,
                       expansion=[], rrf_input='docs'),
    },
    {
        'offset': 5,
        'suffix': 'L.-O--.-.-',
        'fields': ['ocr'],
        'hybrid': False,
        'kwargs': dict(models=['bm25'], bm25_tuned=True,
                       expansion=[], rrf_input='docs'),
    },
    {
        'offset': 6,
        'suffix': 'L.---S.-.-',
        'fields': ['summary'],
        'hybrid': False,
        'kwargs': dict(models=['bm25'], bm25_tuned=True,
                       expansion=[], rrf_input='docs'),
    },
    {
        'offset': 7,
        'suffix': 'L.T---.-.-',
        'fields': ['title'],
        'hybrid': False,
        'kwargs': dict(models=['bm25'], bm25_tuned=True,
                       expansion=[], rrf_input='docs'),
    },
    {
        'offset': 8,
        'suffix': 'L.--F-.-.-',
        'fields': ['folderlabel'],
        'hybrid': False,
        'kwargs': dict(models=['bm25'], bm25_tuned=True,
                       expansion=[], rrf_input='docs'),
    },
]


def run_config(config, query_field):
    """Execute one CONFIGS entry for one query field (T / TD / TDN)."""
    run_folder_name = f"U5.{QUERY_FIELD_TAGS[query_field]}.{config['suffix']}"
    fields = config['fields']

    print(f"\n{Style.BOLD}{Style.CYAN}=== {run_folder_name} "
          f"(query_field={query_field}) ==={Style.RESET}")

    gen_A = RunGenerator(
        searching_fields=[fields],
        query_fields=[query_field],
        **config['kwargs'],
    )

    if config['hybrid']:
        gen_B, sf_B = make_allfl_bm25(query_field)
        run_hybrid(gen_A, fields, gen_B, sf_B, query_field, run_folder_name)
    else:
        run_standard(gen_A, fields, query_field, run_folder_name)


# ---------------------------------------------------------------------------
# Registry & Entry Point
# ---------------------------------------------------------------------------

def _build_registry():
    """id -> (run_folder_name, callable)."""
    registry = {}
    for query_field, base in QUERY_FIELD_ID_BASE.items():
        for config in CONFIGS:
            exp_id = base + config['offset']
            name = f"U5.{QUERY_FIELD_TAGS[query_field]}.{config['suffix']}"
            registry[exp_id] = (
                name,
                (lambda c=config, qf=query_field: run_config(c, qf)),
            )
    return registry


EXPERIMENTS = _build_registry()


def _parse_args(argv):
    """Accepts experiment ids (1, 13, ...) and/or query fields (T, TD, TDN)."""
    if not argv:
        return sorted(EXPERIMENTS.keys())

    selected = []
    for arg in argv:
        if arg.isdigit():
            selected.append(int(arg))
        elif arg.upper() in QUERY_FIELD_ID_BASE:
            base = QUERY_FIELD_ID_BASE[arg.upper()]
            selected.extend(base + c['offset'] for c in CONFIGS)
        else:
            print(f"{Style.WARNING}Warning: unrecognised argument '{arg}', ignoring.{Style.RESET}")
    return sorted(set(selected))


if __name__ == "__main__":
    selected = _parse_args(sys.argv[1:])

    print(f"{Style.BOLD}{Style.CYAN}Running {len(selected)} experiment(s): {selected}{Style.RESET}")
    for exp_id in selected:
        if exp_id in EXPERIMENTS:
            print(f"  {exp_id:>3}  {EXPERIMENTS[exp_id][0]}")

    for exp_id in selected:
        if exp_id not in EXPERIMENTS:
            print(f"{Style.WARNING}Warning: experiment {exp_id} not found, skipping.{Style.RESET}")
            continue
        name, fn = EXPERIMENTS[exp_id]
        try:
            fn()
            print(f"{Style.GREEN}> Experiment {exp_id} ({name}) complete.{Style.RESET}\n")
        except Exception as e:
            print(f"{Style.FAIL}> Experiment {exp_id} ({name}) failed: {e}{Style.RESET}\n")
            raise

    print(f"\n{Style.BOLD}{Style.GREEN}All done!{Style.RESET}")
