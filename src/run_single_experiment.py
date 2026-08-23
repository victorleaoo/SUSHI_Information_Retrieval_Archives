import os
from tqdm import tqdm

from run_generator import RunGenerator, RANDOM_SEED_LIST, RESULTS_PATH, Style

# ---------------------------------------------------------------------------
# Field sets
# ---------------------------------------------------------------------------

TOFS_FIELDS  = ['title', 'ocr', 'folderlabel', 'summary']
TOF_FIELDS   = ['title', 'ocr', 'folderlabel']
TFS_FIELDS   = ['title', 'folderlabel', 'summary']
ALLFL_FIELDS = ['folderlabel']

# Query code → notation segment mapping
QUERY_NOTATION = {'T': 'T--', 'TD': 'TD-', 'TDN': 'TDN'}

def make_output_folder(name: str) -> str:
    path = os.path.abspath(f'../all_runs/{name}')
    os.makedirs(path, exist_ok=True)
    return path


def run_standard(gen, search_fields, query_field, run_folder_name):
    """Run a standard random experiment (30 seeds) with a custom output folder name.

    Only use this for run_type='random'. For 'official_ecf' or 'all_documents',
    use run_once() instead — those are single-shot and must go through
    gen.run_experiments() which has the correct branching logic.
    """
    print(f"\n{Style.BOLD}{Style.GREEN}> Running: {run_folder_name}{Style.RESET}")
    metrics_folder = make_output_folder(run_folder_name)

    if gen.all_folders_folder_label:
        # ALLFL training data is built purely from folderMetadata — it is
        # completely seed-independent. Running 30 seeds would produce identical
        # results every time, so run just once.
        results = gen.run_single_seed(0, search_fields, query_field)
        gen.evaluator.save_run_file(results, RESULTS_PATH, 'AllFolderLabel')
        json_path = os.path.join(metrics_folder, 'AllFolderLabel_TopicsFolderMetrics.json')
        gen.evaluator.evaluate(RESULTS_PATH, json_path)
        gen.evaluator.generate_aggregated_metrics(metrics_folder, 'all_folder_label')
        return

    for seed in tqdm(RANDOM_SEED_LIST, desc=f"({run_folder_name})"):
        results = gen.run_single_seed(seed, search_fields, query_field)
        run_name = f'45-Topics-Random-{seed}'
        gen.evaluator.save_run_file(results, RESULTS_PATH, run_name)
        json_path = os.path.join(metrics_folder, f'Random{seed}_TopicsFolderMetrics.json')
        gen.evaluator.evaluate(RESULTS_PATH, json_path)

    print(f"> Generating aggregated metrics in {metrics_folder}...")
    gen.evaluator.generate_aggregated_metrics(metrics_folder, 'random')


def run_once(gen):
    """Run a single-shot experiment (official_ecf or all_documents).

    Delegates directly to gen.run_experiments(), which already handles both
    run types correctly — no seed loop, trains once (or once per ECF set for
    official_ecf) and evaluates once. Output folder naming is also handled
    internally by saving_folder_name().
    """
    gen.run_experiments()


def single_exp():
    print(f"\n{Style.BOLD}{Style.CYAN}=== EXP: U5.TD-.B.TOFS.-.- ==={Style.RESET}")
    gen = RunGenerator(
        searching_fields=[TOFS_FIELDS], query_fields=['TD'], run_type='official_ecf',
        models=['bm25'], bm25_tuned=False,
        expansion=[], rrf_input='docs',
    )

    if gen.run_type == 'random':
        run_standard(gen, TOFS_FIELDS, 'TD', 'U5.TD-.B.TOFS.-.-')
    else:
        # official_ecf and all_documents are single-shot runs handled by
        # run_experiments() — no seed iteration needed.
        run_once(gen)

single_exp()