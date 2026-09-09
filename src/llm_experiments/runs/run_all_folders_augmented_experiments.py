"""
run_all_folders_augmented_experiments.py
-----------------------------------------
Re-runs the "all folders" (ALLFL) CONFIGS entries from run_new_experiments.py --
the ones whose suffix ends in '.b' or '.c', i.e. every config that touches the
ALLFL ("all folders label") ranker, either as the hybrid RRF partner (offsets
1, 2, 14, 15) or as the ranker itself (offsets 9, 16, 17 -- the ColBERT/BM25/
Embeddings ALLFL baselines) -- for every query field (T, TD, TDN), in two
query variants:

    STD  - the plain, un-augmented query (title/description/narrative only)
    CT   - the plain query plus the topic's core_themes + related_concepts
           text, from the query_expansion/core_themes_and_related_concepts
           LLM experiment
           (data/llm_calls/query_expansion/2_core_themes_and_related_concepts/{T,TD,TDN}.json)

In BOTH variants, the ALLFL ranker's folder-level content is the *augmented*
folder label: the original folderMetadata label (label_parent_expanded +
scope_truncated) plus the folder's core_themes + related_concepts text from
the folder_label_augmentation LLM experiment
(data/llm_calls/folder_label_augmentation/2_folder_context/folders.json) --
never just the plain original label. That's a fixed property of every run in
this script, not a 3rd axis: it's what distinguishes these runs from the
plain ALLFL runs already in all_runs/ produced by run_new_experiments.py.

7 configs x 3 query fields x 2 variants = 42 experiments, all under the
'random' protocol (30 seeds from RANDOM_SEED_LIST), U5 (default
docs_per_box=5, uniform sampling) only -- no DOCSBOX sweep, no OfficialECF
re-runs.

Two of the 7 configs (offsets 16 and 17) are new ALLFL-only baselines added to
CONFIGS in run_new_experiments.py alongside this script: an untuned-BM25 ALLFL
baseline ('-.----.-.b') and an Embeddings ALLFL baseline ('-.----.-.e'),
mirroring the pre-existing ColBERT ALLFL baseline (offset 9, '-.----.-.c').
They're also available un-augmented via run_new_experiments.py directly.

See runs_all_folders.md in this folder for the run-tracking table.

Prerequisites (both already generated):
    data/llm_calls/query_expansion/2_core_themes_and_related_concepts/{T,TD,TDN}.json
    data/llm_calls/folder_label_augmentation/2_folder_context/folders.json
This script does not generate either of them.

Naming: run folder names keep the plain U5.<QF_TAG>.<suffix> pattern for the
STD variant, and use the same 4th-slot query-tag trick as
run_query_augmentated_experiments.py for the CT variant (dropping the base
tag's last char and appending 'CT'). Every folder name additionally gets a
trailing '.FLAug' suffix marking "ALLFL content is the augmented folder
label", so these runs never collide with the plain ALLFL runs already in
all_runs/:
    U5.T--.V.TOFS.mx--2.b.FLAug     (STD)
    U5.T--CT.V.TOFS.mx--2.b.FLAug   (CT)
    U5.TDN.-.----.-.e.FLAug         (STD, new Embeddings ALLFL baseline)
    U5.TDNCT.-.----.-.e.FLAug       (CT, new Embeddings ALLFL baseline)

Known limitation: ColBERT query truncation. All 7 configs involve ColBERT
(as the main ranker in 1/2, as the sole ranker in 9, as the hybrid partner in
14/15) except the two new BM25/Embeddings-only baselines (16, 17). Per the
precedent set in runs.md for the doc_folder_hip DOC/AUG variants, the CT
variant's augmentation text (a core_themes paragraph + related_concepts list)
is likely to be mostly or entirely truncated by ColBERT's ~32-token query
limit. Run these configs anyway rather than excluding them -- a
close-to-null result for a ColBERT-containing config's CT row may reflect
truncation, not a finding about the augmentation itself.

Run from the project root:
    python -m src.llm_experiments.runs.run_all_folders_augmented_experiments              # all 42
    python -m src.llm_experiments.runs.run_all_folders_augmented_experiments 2001 2002    # specific ids
    python -m src.llm_experiments.runs.run_all_folders_augmented_experiments T TD         # every config/variant for those query fields
    python -m src.llm_experiments.runs.run_all_folders_augmented_experiments CT           # every config/query-field for that variant only
"""

import os
import sys

# run_generator.py / run_new_experiments.py use bare imports (e.g. `from
# run_generator import ...`) and expect to run with /src on sys.path, per
# run_new_experiments.py's own docstring ("Run from the /src directory").
# This script instead runs as `python -m src.llm_experiments.runs....` from
# the project root, so /src is not on sys.path by default -- add it here.
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from run_generator import RunGenerator, Style  # noqa: E402
from run_new_experiments import (  # noqa: E402
    CONFIGS,
    QUERY_FIELD_TAGS,
    make_allfl_bm25,
    make_allfl_colbert,
    run_standard,
    run_hybrid,
    is_experiment_done,
)

# The "all folders" configs: every CONFIGS offset whose suffix ends in '.b' or
# '.c' (touches the ALLFL ranker, as hybrid partner or directly).
ALL_FOLDERS_CONFIG_OFFSETS = (1, 2, 9, 14, 15, 16, 17)
ALL_FOLDERS_CONFIGS = [c for c in CONFIGS if c['offset'] in ALL_FOLDERS_CONFIG_OFFSETS]

QUERY_VARIANTS = ('STD', 'CT')

# Query field -> 4-char padded tag; the CT-augmented tag drops the last
# character and appends 'CT' (see module docstring for examples). Mirrors
# run_query_augmentated_experiments.py's QUERY_FIELD_AUG_BASE_TAGS.
QUERY_FIELD_AUG_BASE_TAGS = {'T': 'T---', 'TD': 'TD--', 'TDN': 'TDN-'}


def run_folder_tag(query_field: str, variant: str) -> str:
    if variant == 'STD':
        return QUERY_FIELD_TAGS[query_field]
    return QUERY_FIELD_AUG_BASE_TAGS[query_field][:-1] + variant


# id = 2000 + query_field_offset + variant_offset + config_offset
QUERY_FIELD_ID_OFFSET = {'T': 0, 'TD': 100, 'TDN': 200}
VARIANT_ID_OFFSET = {'STD': 0, 'CT': 30}
ID_BASE = 2000


def run_all_folders_config(config, query_field, variant):
    """Execute one ALL_FOLDERS_CONFIGS entry for one (query_field, query variant) pair."""
    run_folder_name = f"U5.{run_folder_tag(query_field, variant)}.{config['suffix']}.FLAug"
    fields = config['fields']
    query_augmentation = None if variant == 'STD' else variant

    print(f"\n{Style.BOLD}{Style.CYAN}=== {run_folder_name} "
          f"(query_field={query_field}, variant={variant}) ==={Style.RESET}")

    gen_A = RunGenerator(
        searching_fields=[fields],
        query_fields=[query_field],
        query_augmentation=query_augmentation,
        all_folders_folder_label_augmented=True,
        **config['kwargs'],
    )

    if config['hybrid']:
        # The ALLFL partner ranker searches with the same query variant as the
        # main ranker, and always uses the augmented folder label -- both are
        # properties of this script's runs, not of one particular ranker in
        # the fusion.
        if config.get('hybrid_partner', 'bm25') == 'colbert':
            gen_B, sf_B = make_allfl_colbert(
                query_field, query_augmentation=query_augmentation,
                all_folders_folder_label_augmented=True)
        else:
            gen_B, sf_B = make_allfl_bm25(
                query_field, query_augmentation=query_augmentation,
                all_folders_folder_label_augmented=True)
        run_hybrid(gen_A, fields, gen_B, sf_B, query_field, run_folder_name)
    else:
        run_standard(gen_A, fields, query_field, run_folder_name)


def _build_registry():
    """id -> (run_folder_name, callable)."""
    registry = {}
    for query_field, qf_offset in QUERY_FIELD_ID_OFFSET.items():
        for variant, v_offset in VARIANT_ID_OFFSET.items():
            for config in ALL_FOLDERS_CONFIGS:
                exp_id = ID_BASE + qf_offset + v_offset + config['offset']
                name = f"U5.{run_folder_tag(query_field, variant)}.{config['suffix']}.FLAug"
                registry[exp_id] = (
                    name,
                    (lambda c=config, qf=query_field, v=variant: run_all_folders_config(c, qf, v)),
                )
    return registry


EXPERIMENTS = _build_registry()


def _parse_args(argv):
    """Accepts experiment ids (2001, 2032, ...), query fields (T/TD/TDN), and
    query variants (STD/CT)."""
    if not argv:
        return sorted(EXPERIMENTS.keys())

    selected = []
    for arg in argv:
        if arg.isdigit():
            selected.append(int(arg))
        elif arg.upper() in QUERY_FIELD_ID_OFFSET:
            qf = arg.upper()
            for v_offset in VARIANT_ID_OFFSET.values():
                selected.extend(ID_BASE + QUERY_FIELD_ID_OFFSET[qf] + v_offset + c['offset']
                                 for c in ALL_FOLDERS_CONFIGS)
        elif arg.upper() in VARIANT_ID_OFFSET:
            variant = arg.upper()
            for qf_offset in QUERY_FIELD_ID_OFFSET.values():
                selected.extend(ID_BASE + qf_offset + VARIANT_ID_OFFSET[variant] + c['offset']
                                 for c in ALL_FOLDERS_CONFIGS)
        else:
            print(f"{Style.WARNING}Warning: unrecognised argument '{arg}', ignoring.{Style.RESET}")
    return sorted(set(selected))


if __name__ == "__main__":
    selected = _parse_args(sys.argv[1:])

    print(f"{Style.BOLD}{Style.CYAN}Running {len(selected)} experiment(s): {selected}{Style.RESET}")
    for exp_id in selected:
        if exp_id in EXPERIMENTS:
            print(f"  {exp_id:>4}  {EXPERIMENTS[exp_id][0]}")

    for exp_id in selected:
        if exp_id not in EXPERIMENTS:
            print(f"{Style.WARNING}Warning: experiment {exp_id} not found, skipping.{Style.RESET}")
            continue
        name, fn = EXPERIMENTS[exp_id]

        if is_experiment_done(name):
            print(f"{Style.WARNING}> Experiment {exp_id} ({name}) already exists in "
                  f"all_runs, skipping.{Style.RESET}")
            continue

        try:
            fn()
            print(f"{Style.GREEN}> Experiment {exp_id} ({name}) complete.{Style.RESET}\n")
        except Exception as e:
            print(f"{Style.FAIL}> Experiment {exp_id} ({name}) failed: {e}{Style.RESET}\n")
            raise

    print(f"\n{Style.BOLD}{Style.GREEN}All done!{Style.RESET}")
