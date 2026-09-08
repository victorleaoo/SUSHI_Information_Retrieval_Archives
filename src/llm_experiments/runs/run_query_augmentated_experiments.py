"""
run_query_augmentated_experiments.py
-------------------------------------
Re-runs all 15 core CONFIGS from run_new_experiments.py, for every query field
(T, TD, TDN), against three query-augmentation variants built from the
doc_folder_hip hypothetical-document / hypothetical-folder-label LLM outputs
(data/llm_calls/query_expansion/1_doc_folder_hip/{T,TD,TDN}.json):

    DOC  - original query + the three hypothetical documents (doc_1/doc_2/doc_3)
    FL   - original query + the hypothetical folder label (label_text + subject_terms)
    AUG  - original query + DOC text + FL text combined

15 configs x 3 query fields x 3 variants = 135 experiments, all under the
'random' protocol (30 seeds from RANDOM_SEED_LIST), U5 (default docs_per_box=5,
uniform sampling) only -- no DOCSBOX sweep, no OfficialECF re-runs.

See runs.md in this folder for the full rationale, the doc_folder_hip data
format, and a run-tracking table.

Prerequisite: data/llm_calls/query_expansion/1_doc_folder_hip/{T,TD,TDN}.json
must already exist (generated via
`python -m src.llm_experiments.query_expansion.doc_folder_hip.generate`).
This script does not generate them.

Naming: the query-field tag gets a 4th "slot" character that the augmentation
variant overwrites:
    T---   -> T--DOC / T--FL / T--AUG
    TD--   -> TD-DOC / TD-FL / TD-AUG
    TDN-   -> TDNDOC / TDNFL / TDNAUG
e.g. folder U5.T--DOC.V.TOFS.mx--2.b, U5.TD-FL.B.TOFS.-.-, U5.TDNAUG.W.TOFS.-.c

Run from the project root:
    python -m src.llm_experiments.runs.run_query_augmentated_experiments              # all 135
    python -m src.llm_experiments.runs.run_query_augmentated_experiments 1001 1002    # specific ids
    python -m src.llm_experiments.runs.run_query_augmentated_experiments T TD         # every config/variant for those query fields
    python -m src.llm_experiments.runs.run_query_augmentated_experiments DOC          # every config/query-field for that variant only
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
    make_allfl_bm25,
    make_allfl_colbert,
    run_standard,
    run_hybrid,
    is_experiment_done,
)

AUGMENTATION_VARIANTS = ('DOC', 'FL', 'AUG')

# Query field -> 4-char padded tag; the augmented tag drops the last
# character and appends the variant (see module docstring for examples).
QUERY_FIELD_AUG_BASE_TAGS = {'T': 'T---', 'TD': 'TD--', 'TDN': 'TDN-'}


def augmented_tag(query_field: str, variant: str) -> str:
    return QUERY_FIELD_AUG_BASE_TAGS[query_field][:-1] + variant


# id = 1000 + query_field_offset + variant_offset + config_offset(1..15)
QUERY_FIELD_ID_OFFSET = {'T': 0, 'TD': 100, 'TDN': 200}
VARIANT_ID_OFFSET = {'DOC': 0, 'FL': 15, 'AUG': 30}
ID_BASE = 1000


def run_augmented_config(config, query_field, variant):
    """Execute one CONFIGS entry for one (query_field, augmentation variant) pair."""
    run_folder_name = f"U5.{augmented_tag(query_field, variant)}.{config['suffix']}"
    fields = config['fields']

    print(f"\n{Style.BOLD}{Style.CYAN}=== {run_folder_name} "
          f"(query_field={query_field}, augmentation={variant}) ==={Style.RESET}")

    gen_A = RunGenerator(
        searching_fields=[fields],
        query_fields=[query_field],
        query_augmentation=variant,
        **config['kwargs'],
    )

    if config['hybrid']:
        # The ALLFL partner ranker searches with the same augmented query as
        # the main ranker -- the augmentation is a property of the topic's
        # query, not of one particular ranker in the fusion.
        if config.get('hybrid_partner', 'bm25') == 'colbert':
            gen_B, sf_B = make_allfl_colbert(query_field, query_augmentation=variant)
        else:
            gen_B, sf_B = make_allfl_bm25(query_field, query_augmentation=variant)
        run_hybrid(gen_A, fields, gen_B, sf_B, query_field, run_folder_name)
    else:
        run_standard(gen_A, fields, query_field, run_folder_name)


def _build_registry():
    """id -> (run_folder_name, callable)."""
    registry = {}
    for query_field, qf_offset in QUERY_FIELD_ID_OFFSET.items():
        for variant, v_offset in VARIANT_ID_OFFSET.items():
            for config in CONFIGS:
                exp_id = ID_BASE + qf_offset + v_offset + config['offset']
                name = f"U5.{augmented_tag(query_field, variant)}.{config['suffix']}"
                registry[exp_id] = (
                    name,
                    (lambda c=config, qf=query_field, v=variant: run_augmented_config(c, qf, v)),
                )
    return registry


EXPERIMENTS = _build_registry()


def _parse_args(argv):
    """Accepts experiment ids (1001, 1032, ...), query fields (T/TD/TDN), and
    augmentation variants (DOC/FL/AUG)."""
    if not argv:
        return sorted(EXPERIMENTS.keys())

    selected = []
    for arg in argv:
        if arg.isdigit():
            selected.append(int(arg))
        elif arg.upper() in QUERY_FIELD_ID_OFFSET:
            qf = arg.upper()
            for v_offset in VARIANT_ID_OFFSET.values():
                selected.extend(ID_BASE + QUERY_FIELD_ID_OFFSET[qf] + v_offset + c['offset'] for c in CONFIGS)
        elif arg.upper() in VARIANT_ID_OFFSET:
            variant = arg.upper()
            for qf_offset in QUERY_FIELD_ID_OFFSET.values():
                selected.extend(ID_BASE + qf_offset + VARIANT_ID_OFFSET[variant] + c['offset'] for c in CONFIGS)
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
