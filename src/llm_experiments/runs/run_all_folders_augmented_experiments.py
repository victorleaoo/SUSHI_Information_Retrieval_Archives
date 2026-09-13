"""
run_all_folders_augmented_experiments.py
-----------------------------------------
Re-runs every CONFIGS entry from run_new_experiments.py whose fields include
the folder-label field ('F' in the TOFS code), for every query field
(T, TD, TDN), in two query variants:

    STD  - the plain, un-augmented query (title/description/narrative only)
    CT   - the plain query plus the topic's core_themes + related_concepts
           text, from the query_expansion/core_themes_and_related_concepts
           LLM experiment
           (data/llm_calls/query_expansion/2_core_themes_and_related_concepts/{T,TD,TDN}.json)

In BOTH variants, every folder-label field touched by these runs -- whether
it's the ALLFL ("all folders label") ranker's content or the main ranker's
own per-document 'folderlabel' field -- uses the *augmented* label: the
original folderMetadata label (label_parent_expanded + scope_truncated) plus
the folder's core_themes + related_concepts text from the
folder_label_augmentation LLM experiment
(data/llm_calls/folder_label_augmentation/2_folder_context/folders.json).
That's a fixed property of every run in this script, not a 3rd axis.

Three groups of configs, by how the folder-label field appears in them:

  Group A -- hybrid configs whose main ranker also searches 'folderlabel'
  directly (it's part of TOFS_FIELDS) AND fuse in a separate ALLFL ranker as
  hybrid partner B (offsets 1, 2, 14, 15: 'V', 'X', 'W.TOFS.-.c',
  'W.TOFS.s---2.c'). Both sides now get the augmented label: the main
  ranker via RunGenerator(folder_label_augmented=True), the ALLFL partner via
  all_folders_folder_label_augmented=True (as before). Older FLAug runs of
  these same 4 configs already exist in all_runs/ under a plain 'TOFS' fields
  code -- those only had the ALLFL partner augmented, not the main ranker's
  own F field, so they are a DIFFERENT (weaker) run and are deliberately left
  in place rather than overwritten. These re-runs use a distinct name: the
  fields code has 'F' replaced with 'FAUG' (TOFS -> TOFAUGS), keeping the
  trailing '.FLAug' marker since an ALLFL ranker using FLAug is still present:
      U5.T--.V.TOFAUGS.mx--2.b.FLAug   (STD)
      U5.T--CT.V.TOFAUGS.mx--2.b.FLAug (CT)

  Group B -- standalone (non-hybrid) configs whose only ranker searches
  'folderlabel' as one of its plain fields, with NO separate ALLFL ranker
  involved at all (offsets 3, 4, 8, 10, 11, 12, 13: 'L.TOFS.mx--2.-',
  'B.TOFS.-.-', 'L.--F-.-.-', 'C.TOFS.-.-', 'E.TOFS.-.-', 'Z.TOFS.-.-',
  'W.TOFS.-.-'). These have no prior FLAug run to collide with. The main
  ranker gets RunGenerator(folder_label_augmented=True); the fields code has
  'F' replaced with 'FAUG' same as Group A, but WITHOUT a trailing '.FLAug'
  suffix -- that suffix specifically marks "an ALLFL ranker in this run is
  using FLAug", which doesn't apply here:
      U5.T--.L.TOFAUGS.mx--2.-    (STD)
      U5.T--CT.L.TOFAUGS.mx--2.- (CT)
      U5.T--.L.--FAUG-.-.-        (STD, offset 8)

  Group C -- the pure ALLFL baselines, whose only ranker IS the ALLFL ranker
  (offsets 9, 16, 17: '-.----.-.c', '-.----.-.b', '-.----.-.e'). Fields are
  all dashes -- there's no separate main-ranker F field to further augment.
  Unchanged from the original version of this script:
      U5.TDN.-.----.-.e.FLAug         (STD)
      U5.TDNCT.-.----.-.e.FLAug       (CT)

14 configs (4 + 7 + 3) x 3 query fields x 2 variants = 84 experiments, all
under the 'random' protocol (30 seeds from RANDOM_SEED_LIST), U5 (default
docs_per_box=5, uniform sampling) only -- no DOCSBOX sweep, no OfficialECF
re-runs.

See runs_all_folders.md in this folder for the run-tracking table.

Prerequisites (both already generated):
    data/llm_calls/query_expansion/2_core_themes_and_related_concepts/{T,TD,TDN}.json
    data/llm_calls/folder_label_augmentation/2_folder_context/folders.json
This script does not generate either of them.

Naming: run folder names keep the plain U5.<QF_TAG>.<suffix> pattern for the
STD variant, and use the same 4th-slot query-tag trick as
run_query_augmentated_experiments.py for the CT variant (dropping the base
tag's last char and appending 'CT'). See the group descriptions above for the
fields-code ('F' -> 'FAUG') and trailing '.FLAug' rules.

Known limitation: ColBERT query truncation. Every Group A config and several
Group B configs (C, W, Z, and the ColBERT-only ALLFL baseline) involve
ColBERT. Per the precedent set in runs.md for the doc_folder_hip DOC/AUG
variants, the CT variant's augmentation text (a core_themes paragraph +
related_concepts list) is likely to be mostly or entirely truncated by
ColBERT's ~32-token query limit. Run these configs anyway rather than
excluding them -- a close-to-null result for a ColBERT-containing config's CT
row may reflect truncation, not a finding about the augmentation itself.

Run from the project root:
    python -m src.llm_experiments.runs.run_all_folders_augmented_experiments              # all 84
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

# Group A: hybrid configs -- main ranker searches 'folderlabel' directly AND
# fuses in a separate ALLFL ranker as hybrid partner.
GROUP_A_OFFSETS = (1, 2, 14, 15)

# Group B: standalone configs -- main ranker searches 'folderlabel' directly,
# no separate ALLFL ranker involved.
GROUP_B_OFFSETS = (3, 4, 8, 10, 11, 12, 13)

# Group C: pure ALLFL baselines -- the only ranker IS the ALLFL ranker.
GROUP_C_OFFSETS = (9, 16, 17)

ALL_FOLDERS_CONFIG_OFFSETS = GROUP_A_OFFSETS + GROUP_B_OFFSETS + GROUP_C_OFFSETS
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


def augmented_fields_code(suffix: str) -> str:
    """Replaces 'F' with 'FAUG' in a config suffix's fields-code segment
    (2nd dot-separated slot, e.g. 'TOFS' -> 'TOFAUGS', '--F-' -> '--FAUG-'),
    marking that the main ranker's own folderlabel field is augmented."""
    parts = suffix.split('.')
    parts[1] = parts[1].replace('F', 'FAUG')
    return '.'.join(parts)


def build_run_folder_name(config, query_field, variant):
    offset = config['offset']
    tag = run_folder_tag(query_field, variant)

    if offset in GROUP_C_OFFSETS:
        return f"U5.{tag}.{config['suffix']}.FLAug"
    # Group A and Group B both use the 'FAUG'-coded fields segment; only
    # Group A keeps the trailing '.FLAug' (it still has an ALLFL ranker).
    suffix = augmented_fields_code(config['suffix'])
    if offset in GROUP_A_OFFSETS:
        return f"U5.{tag}.{suffix}.FLAug"
    return f"U5.{tag}.{suffix}"


def run_all_folders_config(config, query_field, variant):
    """Execute one ALL_FOLDERS_CONFIGS entry for one (query_field, query variant) pair."""
    offset = config['offset']
    run_folder_name = build_run_folder_name(config, query_field, variant)
    fields = config['fields']
    query_augmentation = None if variant == 'STD' else variant

    print(f"\n{Style.BOLD}{Style.CYAN}=== {run_folder_name} "
          f"(query_field={query_field}, variant={variant}) ==={Style.RESET}")

    if offset in GROUP_C_OFFSETS:
        # Pure ALLFL baseline: the only ranker IS the ALLFL ranker.
        gen_A = RunGenerator(
            searching_fields=[fields],
            query_fields=[query_field],
            query_augmentation=query_augmentation,
            all_folders_folder_label_augmented=True,
            **config['kwargs'],
        )
        run_standard(gen_A, fields, query_field, run_folder_name)
        return

    # Groups A and B: the main ranker searches 'folderlabel' directly, so its
    # own F field is augmented via folder_label_augmented=True.
    gen_A = RunGenerator(
        searching_fields=[fields],
        query_fields=[query_field],
        query_augmentation=query_augmentation,
        folder_label_augmented=True,
        **config['kwargs'],
    )

    if offset in GROUP_A_OFFSETS:
        # Group A additionally fuses in a separate ALLFL ranker as hybrid
        # partner, also augmented -- both sides of the fusion use FLAug.
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
        # Group B: standalone, no ALLFL ranker involved.
        run_standard(gen_A, fields, query_field, run_folder_name)


def _build_registry():
    """id -> (run_folder_name, callable)."""
    registry = {}
    for query_field, qf_offset in QUERY_FIELD_ID_OFFSET.items():
        for variant, v_offset in VARIANT_ID_OFFSET.items():
            for config in ALL_FOLDERS_CONFIGS:
                exp_id = ID_BASE + qf_offset + v_offset + config['offset']
                name = build_run_folder_name(config, query_field, variant)
                registry[exp_id] = (
                    name,
                    (lambda c=config, qf=query_field, v=variant: run_all_folders_config(c, qf, v)),
                )
    return registry


# id = 2000 + query_field_offset + variant_offset + config_offset
QUERY_FIELD_ID_OFFSET = {'T': 0, 'TD': 100, 'TDN': 200}
VARIANT_ID_OFFSET = {'STD': 0, 'CT': 30}
ID_BASE = 2000

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
