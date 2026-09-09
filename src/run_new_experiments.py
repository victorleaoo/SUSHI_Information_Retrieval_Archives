"""
run_new_experiments.py
----------------------
Full re-run of the 17 core configurations for EVERY query field (T, TD, TDN),
plus the docs-per-box sweep (U1-U4/K5/A-) and the official-ECF re-runs of
configs 1-8.

17 configurations x 3 query fields = 51 experiments, all under the 'random'
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
    U5.<QF>.-.----.-.c      - ALLFL ColBERT baseline (no main-ranker fields)
    U5.<QF>.C.TOFS.-.-      - ColBERT only, TOFS
    U5.<QF>.E.TOFS.-.-      - Embeddings only, TOFS
    U5.<QF>.Z.TOFS.-.-      - Z (untuned BM25F + Embeddings + ColBERT), TOFS, no hybrid
    U5.<QF>.W.TOFS.-.-      - W (tuned BM25F + Embeddings + ColBERT), TOFS, no hybrid
    U5.<QF>.W.TOFS.-.c      - W, TOFS, hybrid RRF with the ALLFL ColBERT ranker
    U5.<QF>.W.TOFS.s---2.c  - W, TOFS, similar_snc k=2 expansion, hybrid RRF with
                              the ALLFL ColBERT ranker
    U5.<QF>.-.----.-.b      - ALLFL untuned-BM25 baseline (no main-ranker fields)
    U5.<QF>.-.----.-.e      - ALLFL Embeddings baseline (no main-ranker fields)

Experiment ids (core 17x3 = 51):
    T   ->  1..17     (folders U5.T--.*)
    TD  -> 31..47     (folders U5.TD-.*)
    TDN -> 61..77     (folders U5.TDN.*)

Docs-per-box sweep (6 variants x 3 query fields = 18), all using the 'W'
config (tuned BM25F + Embeddings + ColBERT, TOFS, no expansion/hybrid):
    U1  - uniform sampling, 1 doc/box    U2  - uniform, 2 docs/box
    U3  - uniform, 3 docs/box            U4  - uniform, 4 docs/box
    K5  - uneven sampling (5-per-box-equivalent target distribution)
    A-  - 'all_documents' oracle run (every document, single pass)
    ids:  T -> 101..106   TD -> 111..116   TDN -> 121..126
    (order within each block: U1, U2, U3, U4, K5, A-)

Official-ECF re-runs (offsets 1..8 of CONFIGS, 8 x 3 query fields = 24),
using the NTCIR-18 official protocol (3 ExperimentSets, no random seeds)
instead of the 30-seed random protocol:
    ids:  T -> 201..208   TD -> 231..238   TDN -> 261..268
    (folders get an '.OfficialECF' suffix, e.g. U5.TD-.B.TOFS.-.-.OfficialECF)

Run from the /src directory:
    python run_new_experiments.py              # Run ALL experiments (core + sweep + official)
    python run_new_experiments.py 1 3 5        # Run specific experiment ids
    python run_new_experiments.py T TDN        # Run every core config for those query fields
    python run_new_experiments.py DOCSBOX      # Run the full U1-U4/K5/A- sweep (all query fields)
    python run_new_experiments.py OFFICIAL     # Run the offset 1-8 official-ECF re-runs (all query fields)
"""

import os
import sys
from tqdm import tqdm

from run_generator import RunGenerator, RANDOM_SEED_LIST, RESULTS_PATH, Style, PROJECT_ROOT
from hybrid_models import perform_hybrid_fusion

# ---------------------------------------------------------------------------
# Field sets & query fields
# ---------------------------------------------------------------------------

TOFS_FIELDS  = ['title', 'ocr', 'folderlabel', 'summary']
ALLFL_FIELDS = ['folderlabel']

# Query field -> tag used in the run folder name (2nd slot, always 3 chars).
QUERY_FIELD_TAGS = {'T': 'T--', 'TD': 'TD-', 'TDN': 'TDN'}

# Base id offset per query field: T -> 1..15, TD -> 31..45, TDN -> 61..75
QUERY_FIELD_ID_BASE = {'T': 0, 'TD': 30, 'TDN': 60}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_output_folder(name: str) -> str:
    path = os.path.join(PROJECT_ROOT, 'all_runs', name)
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


def make_allfl_bm25(query_field, query_augmentation=None, all_folders_folder_label_augmented=False):
    """Shared ALLFL BM25 RunGenerator used as ranker B in the hybrid experiments."""
    gen = RunGenerator(
        searching_fields=[ALLFL_FIELDS],
        query_fields=[query_field],
        models=['bm25'],
        bm25_tuned=False,
        expansion=[],
        all_folders_folder_label=True,
        all_folders_folder_label_augmented=all_folders_folder_label_augmented,
        query_augmentation=query_augmentation,
    )
    return gen, ALLFL_FIELDS


def make_allfl_colbert(query_field, query_augmentation=None, all_folders_folder_label_augmented=False):
    """Shared ALLFL ColBERT RunGenerator used as ranker B in the colbert-hybrid experiments."""
    gen = RunGenerator(
        searching_fields=[ALLFL_FIELDS],
        query_fields=[query_field],
        models=['colbert'],
        expansion=[],
        all_folders_folder_label=True,
        all_folders_folder_label_augmented=all_folders_folder_label_augmented,
        query_augmentation=query_augmentation,
    )
    return gen, ALLFL_FIELDS


def run_all_documents(gen, search_fields, query_field, run_folder_name):
    """Run a single-shot 'all_documents' (oracle) experiment with a custom output folder name."""
    print(f"\n{Style.BOLD}{Style.GREEN}> Running (All Documents): {run_folder_name}{Style.RESET}")
    metrics_folder = make_output_folder(run_folder_name)

    results = gen.run_single_seed(0, search_fields, query_field)
    run_name = '45-Topics-AllDocuments'
    gen.evaluator.save_run_file(results, RESULTS_PATH, run_name)
    json_path = os.path.join(metrics_folder, 'AllDocuments_TopicsFolderMetrics.json')
    gen.evaluator.evaluate(RESULTS_PATH, json_path)

    print(f"> Generating aggregated metrics in {metrics_folder}...")
    gen.evaluator.generate_aggregated_metrics(metrics_folder, 'all_documents')


def run_official(gen, search_fields, query_field, run_folder_name):
    """Run a standard (non-hybrid) experiment under the NTCIR-18 official ECF protocol."""
    print(f"\n{Style.BOLD}{Style.GREEN}> Running (Official ECF): {run_folder_name}{Style.RESET}")
    metrics_folder = make_output_folder(run_folder_name)

    gen.run_type = 'official_ecf'
    all_results = gen.run_official_ecf(search_fields, query_field)

    run_name = 'OfficialECF-3Sets-45Topics'
    gen.evaluator.save_run_file(all_results, RESULTS_PATH, run_name)
    json_path = os.path.join(metrics_folder, 'OfficialECF_TopicsFolderMetrics.json')
    gen.evaluator.evaluate(RESULTS_PATH, json_path)

    print(f"> Generating aggregated metrics in {metrics_folder}...")
    gen.evaluator.generate_aggregated_metrics(metrics_folder, 'official_ecf')


def run_hybrid_official(gen_A, search_field_A, gen_B, search_field_B, query_field, run_folder_name):
    """Run a hybrid experiment (document ranker + ALLFL ranker RRF) under the official ECF protocol."""
    print(f"\n{Style.BOLD}{Style.GREEN}> Running Hybrid (Official ECF): {run_folder_name}{Style.RESET}")
    metrics_folder = make_output_folder(run_folder_name)

    gen_A.run_type = 'official_ecf'
    gen_B.run_type = 'official_ecf'
    res_A = gen_A.run_official_ecf(search_field_A, query_field)
    res_B = gen_B.run_official_ecf(search_field_B, query_field)
    final = perform_hybrid_fusion(res_A, res_B)

    run_name = 'OfficialECF-3Sets-45Topics'
    gen_A.evaluator.save_run_file(final, RESULTS_PATH, run_name)
    json_path = os.path.join(metrics_folder, 'OfficialECF_TopicsFolderMetrics.json')
    gen_A.evaluator.evaluate(RESULTS_PATH, json_path)

    print(f"> Generating aggregated metrics in {metrics_folder}...")
    gen_A.evaluator.generate_aggregated_metrics(metrics_folder, 'official_ecf')


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
    {
        'offset': 9,
        'suffix': '-.----.-.c',
        'fields': ALLFL_FIELDS,
        'hybrid': False,
        'kwargs': dict(models=['colbert'], expansion=[], all_folders_folder_label=True),
    },
    {
        'offset': 10,
        'suffix': 'C.TOFS.-.-',
        'fields': TOFS_FIELDS,
        'hybrid': False,
        'kwargs': dict(models=['colbert'], expansion=[], rrf_input='docs'),
    },
    {
        'offset': 11,
        'suffix': 'E.TOFS.-.-',
        'fields': TOFS_FIELDS,
        'hybrid': False,
        'kwargs': dict(models=['embeddings'], expansion=[], rrf_input='docs'),
    },
    {
        'offset': 12,
        'suffix': 'Z.TOFS.-.-',
        'fields': TOFS_FIELDS,
        'hybrid': False,
        'kwargs': dict(models=['bm25', 'embeddings', 'colbert'], bm25_tuned=False,
                       expansion=[], rrf_input='docs'),
    },
    {
        'offset': 13,
        'suffix': 'W.TOFS.-.-',
        'fields': TOFS_FIELDS,
        'hybrid': False,
        'kwargs': dict(models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
                       expansion=[], rrf_input='docs'),
    },
    {
        'offset': 14,
        'suffix': 'W.TOFS.-.c',
        'fields': TOFS_FIELDS,
        'hybrid': True,
        'hybrid_partner': 'colbert',
        'kwargs': dict(models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
                       expansion=[], rrf_input='docs'),
    },
    {
        'offset': 15,
        'suffix': 'W.TOFS.s---2.c',
        'fields': TOFS_FIELDS,
        'hybrid': True,
        'hybrid_partner': 'colbert',
        'kwargs': dict(models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
                       expansion=['similar_snc'], expansion_ceiling_k=2, rrf_input='docs'),
    },
    {
        'offset': 16,
        'suffix': '-.----.-.b',
        'fields': ALLFL_FIELDS,
        'hybrid': False,
        'kwargs': dict(models=['bm25'], bm25_tuned=False,
                       expansion=[], all_folders_folder_label=True),
    },
    {
        'offset': 17,
        'suffix': '-.----.-.e',
        'fields': ALLFL_FIELDS,
        'hybrid': False,
        'kwargs': dict(models=['embeddings'], expansion=[], all_folders_folder_label=True),
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
        if config.get('hybrid_partner', 'bm25') == 'colbert':
            gen_B, sf_B = make_allfl_colbert(query_field)
        else:
            gen_B, sf_B = make_allfl_bm25(query_field)
        run_hybrid(gen_A, fields, gen_B, sf_B, query_field, run_folder_name)
    else:
        run_standard(gen_A, fields, query_field, run_folder_name)


def run_config_official(config, query_field):
    """Execute one of the offset-1..8 CONFIGS entries for one query field under
    the NTCIR-18 official ECF protocol (instead of the 30-seed random protocol)."""
    run_folder_name = f"U5.{QUERY_FIELD_TAGS[query_field]}.{config['suffix']}.OfficialECF"
    fields = config['fields']

    print(f"\n{Style.BOLD}{Style.CYAN}=== {run_folder_name} "
          f"(query_field={query_field}) ==={Style.RESET}")

    gen_A = RunGenerator(
        searching_fields=[fields],
        query_fields=[query_field],
        **config['kwargs'],
    )

    if config['hybrid']:
        if config.get('hybrid_partner', 'bm25') == 'colbert':
            gen_B, sf_B = make_allfl_colbert(query_field)
        else:
            gen_B, sf_B = make_allfl_bm25(query_field)
        run_hybrid_official(gen_A, fields, gen_B, sf_B, query_field, run_folder_name)
    else:
        run_official(gen_A, fields, query_field, run_folder_name)


# ---------------------------------------------------------------------------
# Docs-per-box sweep (U1-U4 / K5 / A-)
# ---------------------------------------------------------------------------
# All variants use the 'W' configuration (tuned BM25F + Embeddings + ColBERT,
# TOFS fields, no expansion, no hybrid) — i.e. CONFIGS offset 13 — varying only
# how the training-document ECF is sampled:
#   U<N>  - uniform sampling, N documents per box (N = 1..4; U5 is the default
#           already covered by CONFIGS offset 13 above)
#   K5    - uneven sampling (fixed per-box targets from RGdistribution.xlsx)
#   A-    - 'all_documents' oracle run (every document in the collection, no
#           sampling, single pass instead of 30 seeds)
W_KWARGS = dict(models=['bm25', 'embeddings', 'colbert'], bm25_tuned=True,
                expansion=[], rrf_input='docs')

DOCSBOX_VARIANTS = [
    {'tag': 'U1', 'docs_per_box': 1, 'sampling': 'uniform', 'run_type': 'random'},
    {'tag': 'U2', 'docs_per_box': 2, 'sampling': 'uniform', 'run_type': 'random'},
    {'tag': 'U3', 'docs_per_box': 3, 'sampling': 'uniform', 'run_type': 'random'},
    {'tag': 'U4', 'docs_per_box': 4, 'sampling': 'uniform', 'run_type': 'random'},
    {'tag': 'K5', 'docs_per_box': 5, 'sampling': 'uneven', 'run_type': 'random'},
    {'tag': 'A-', 'docs_per_box': 5, 'sampling': 'uniform', 'run_type': 'all_documents'},
]


def run_docsbox_config(variant, query_field):
    """Execute one DOCSBOX_VARIANTS entry (W.TOFS.-.-) for one query field."""
    run_folder_name = f"{variant['tag']}.{QUERY_FIELD_TAGS[query_field]}.W.TOFS.-.-"

    print(f"\n{Style.BOLD}{Style.CYAN}=== {run_folder_name} "
          f"(query_field={query_field}) ==={Style.RESET}")

    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS],
        query_fields=[query_field],
        docs_per_box=variant['docs_per_box'],
        sampling=variant['sampling'],
        run_type=variant['run_type'],
        **W_KWARGS,
    )

    if variant['run_type'] == 'all_documents':
        run_all_documents(gen, TOFS_FIELDS, query_field, run_folder_name)
    else:
        run_standard(gen, TOFS_FIELDS, query_field, run_folder_name)


# ---------------------------------------------------------------------------
# Registry & Entry Point
# ---------------------------------------------------------------------------

# Docs-per-box sweep ids: 100 + query_field_slot(0/10/20) + variant offset(1..6)
DOCSBOX_ID_BASE = {'T': 100, 'TD': 110, 'TDN': 120}

# Official-ECF re-run ids (offsets 1..8 only): 200 + query_field_slot(0/30/60)
OFFICIAL_ID_BASE = {'T': 200, 'TD': 230, 'TDN': 260}
OFFICIAL_CONFIGS = [c for c in CONFIGS if c['offset'] <= 8]


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

    for query_field, base in DOCSBOX_ID_BASE.items():
        for offset, variant in enumerate(DOCSBOX_VARIANTS, start=1):
            exp_id = base + offset
            name = f"{variant['tag']}.{QUERY_FIELD_TAGS[query_field]}.W.TOFS.-.-"
            registry[exp_id] = (
                name,
                (lambda v=variant, qf=query_field: run_docsbox_config(v, qf)),
            )

    for query_field, base in OFFICIAL_ID_BASE.items():
        for config in OFFICIAL_CONFIGS:
            exp_id = base + config['offset']
            name = f"U5.{QUERY_FIELD_TAGS[query_field]}.{config['suffix']}.OfficialECF"
            registry[exp_id] = (
                name,
                (lambda c=config, qf=query_field: run_config_official(c, qf)),
            )

    return registry


EXPERIMENTS = _build_registry()


def _parse_args(argv):
    """Accepts experiment ids (1, 13, ...), query fields (T, TD, TDN), and the
    group keywords DOCSBOX (the U1-U4/K5/A- sweep) and OFFICIAL (the offset
    1-8 official-ECF re-runs)."""
    if not argv:
        return sorted(EXPERIMENTS.keys())

    selected = []
    for arg in argv:
        if arg.isdigit():
            selected.append(int(arg))
        elif arg.upper() in QUERY_FIELD_ID_BASE:
            base = QUERY_FIELD_ID_BASE[arg.upper()]
            selected.extend(base + c['offset'] for c in CONFIGS)
        elif arg.upper() == 'DOCSBOX':
            for base in DOCSBOX_ID_BASE.values():
                selected.extend(base + offset for offset in range(1, len(DOCSBOX_VARIANTS) + 1))
        elif arg.upper() == 'OFFICIAL':
            for base in OFFICIAL_ID_BASE.values():
                selected.extend(base + c['offset'] for c in OFFICIAL_CONFIGS)
        else:
            print(f"{Style.WARNING}Warning: unrecognised argument '{arg}', ignoring.{Style.RESET}")
    return sorted(set(selected))


DONE_MARKER_FILES = ('model_overall_stats.json', 'all_documents_model_overall_stats.json')


def is_experiment_done(name):
    """
    An experiment is considered already done only if its ../all_runs/<name>
    folder contains a completion marker (the aggregated overall-stats file
    written at the very end of generate_aggregated_metrics).

    NOTE: make_output_folder() creates the output folder up front, before any
    actual computation runs. Checking bare folder existence would therefore
    treat a folder left behind by an interrupted/crashed run as "done" and
    skip it forever, even though it has no results in it. Checking for the
    marker file avoids that trap.
    """
    folder = os.path.join(PROJECT_ROOT, 'all_runs', name)
    return any(os.path.isfile(os.path.join(folder, marker)) for marker in DONE_MARKER_FILES)


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

        if is_experiment_done(name):
            print(f"{Style.WARNING}> Experiment {exp_id} ({name}) already exists in "
                  f"../all_runs, skipping.{Style.RESET}")
            continue

        try:
            fn()
            print(f"{Style.GREEN}> Experiment {exp_id} ({name}) complete.{Style.RESET}\n")
        except Exception as e:
            print(f"{Style.FAIL}> Experiment {exp_id} ({name}) failed: {e}{Style.RESET}\n")
            raise

    print(f"\n{Style.BOLD}{Style.GREEN}All done!{Style.RESET}")
