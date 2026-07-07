import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

"""
Experiment Runner for SNC Document Expansion Baseline.

This script runs document-level retrieval experiments where SNC folder labels
are enriched with document text fields (title, OCR first page, summary)
aggregated at the SNC level from ECF training documents.

PURPOSE:
    These experiments serve as a BASELINE for future SNC LLM expansion experiments.
    Before using an LLM to expand SNC labels, we first measure the impact of adding
    raw document text to see how much document information alone can improve retrieval.

HOW IT WORKS:
    1. For each random seed, an ECF is created (uniform sampling, 5 docs/box).
    2. ECF documents are grouped by their folder's SNC code.
    3. For each SNC, the specified document fields (T/O/S combinations) from all
       documents belonging to folders with that SNC are concatenated.
    4. This aggregated text is appended to the SNC base label (label_parent_expanded
       and/or raw_scope).
    5. ALL folders sharing the same SNC receive the same enriched label — even folders
       without ECF documents.
    6. Folders with SNC='Unknown' fall back to the SNC base field only.

DOCUMENT FIELD COMBINATIONS (TOS):
    - T: title only
    - O: ocr (first page) only
    - S: summary only
    - TO: title + ocr
    - TS: title + summary
    - OS: ocr + summary
    - TOS: title + ocr + summary

SNC BASE FIELD OPTIONS:
    - LP: label_parent_expanded (e.g., "POLITICAL AFFAIRS & RELATIONS: ELECTIONS")
    - LP+RS: label_parent_expanded + raw_scope (adds full scope note text)

TOPIC QUERY CONFIGURATIONS:
    - Standard TD: Title + Description from topics_output.txt
    - Expanded TD+CKS: Title + Description + combined_keyword_string from topics_expanded.txt
    - Expanded TD+DHS: Title + Description + dense_hyde_summary from topics_expanded.txt

MODELS:
    Singles only: BM25, ColBERT, Embeddings (no multi-model RRF).

OUTPUT FOLDER NAMING:
    SNCDE-<SNC_BASE>-<DOC_FIELDS>_NEX_<TOPIC_CONFIG>_<MODEL>
    Examples:
        SNCDE-LP-T_NEX_TD_BM25
        SNCDE-LP-RS-TOS_NEX_EXP-T-D-CKS_COLBERT

USAGE:
    # Run all preset experiments:
    python run_snc_doc_experiments.py

    # Or import and run specific configurations:
    from run_snc_doc_experiments import run_snc_doc_experiment
    run_snc_doc_experiment(
        snc_doc_fields=['title', 'summary'],
        snc_base_fields=['label_parent_expanded'],
        topic_source='standard',
        models=['bm25'],
        description="SNC + Title+Summary | LP base | BM25"
    )
"""

from run_generator import RunGenerator, Style


# ============================================================================
#  CORE EXPERIMENT FUNCTION
# ============================================================================

def run_snc_doc_experiment(
    snc_doc_fields=None,
    snc_base_fields=None,
    topic_source='standard',
    topic_query_fields=None,
    models=None,
    query_fields=None,
    description=""
):
    """
    Runs a single SNC Document Expansion experiment.

    All experiments use snc_doc_expansion=True with uniform sampling and 3 random seeds.

    Args:
        snc_doc_fields (list): Document fields to aggregate per SNC.
            Subset of ['title', 'ocr', 'summary']. Default: ['title', 'ocr', 'summary'].
        snc_base_fields (list): SNC metadata fields for the base label.
            Subset of ['label_parent_expanded', 'raw_scope']. Default: ['label_parent_expanded'].
        topic_source (str): 'standard' or 'expanded'.
        topic_query_fields (list): Fields for query construction (when topic_source='expanded').
        models (list): List of model names. Default: ['bm25'].
        query_fields (list): Standard query field modes (e.g., ['TD']).
        description (str): Human-readable description printed before the run.
    """
    if snc_doc_fields is None:
        snc_doc_fields = ['title', 'ocr', 'summary']
    if snc_base_fields is None:
        snc_base_fields = ['label_parent_expanded']
    if models is None:
        models = ['bm25']
    if query_fields is None:
        query_fields = ['TD']

    print(f"\n{Style.BOLD}{Style.HEADER}{'='*70}{Style.RESET}")
    print(f"{Style.BOLD}{Style.GREEN}> EXPERIMENT: {description}{Style.RESET}")
    print(f"  SNC Doc Fields: {snc_doc_fields}")
    print(f"  SNC Base Fields: {snc_base_fields}")
    print(f"  Topic Source:    {topic_source}")
    print(f"  Topic Fields:    {topic_query_fields}")
    print(f"  Models:          {models}")
    print(f"{Style.BOLD}{Style.HEADER}{'='*70}{Style.RESET}\n")

    gen = RunGenerator(
        searching_fields=[['folderlabel']],
        query_fields=query_fields,
        run_type='random',
        models=models,
        sampling='uniform',
        expansion=[],
        all_folders_folder_label=False,
        rrf_input='folders',
        snc_doc_expansion=True,
        snc_doc_fields=snc_doc_fields,
        snc_base_fields=snc_base_fields,
        topic_source=topic_source,
        topic_query_fields=topic_query_fields,
    )

    gen.run_experiments()


# ============================================================================
#  DOCUMENT FIELD COMBINATIONS
# ============================================================================

# All non-empty subsets of {title, ocr, summary}
TOS_COMBINATIONS = [
    ['title'],
    ['ocr'],
    ['summary'],
    ['title', 'ocr'],
    ['title', 'summary'],
    ['ocr', 'summary'],
    ['title', 'ocr', 'summary'],
]

# Human-readable abbreviations for logging
TOS_NAMES = {
    ('title',): 'T',
    ('ocr',): 'O',
    ('summary',): 'S',
    ('title', 'ocr'): 'TO',
    ('title', 'summary'): 'TS',
    ('ocr', 'summary'): 'OS',
    ('title', 'ocr', 'summary'): 'TOS',
}


# ============================================================================
#  PRESET EXPERIMENT CONFIGURATIONS
# ============================================================================

def _build_presets(snc_base_fields, base_name):
    """
    Builds preset experiment configs for all TOS × topic combinations
    with a given SNC base field configuration.

    Args:
        snc_base_fields (list): The SNC base fields to use.
        base_name (str): Human-readable name for the base (e.g., 'LP', 'LP+RS').

    Returns:
        list: List of experiment config dicts.
    """
    presets = []

    for tos in TOS_COMBINATIONS:
        tos_name = TOS_NAMES[tuple(tos)]

        # Standard Topics (TD)
        presets.append({
            "description": f"SNC+{tos_name} | {base_name} base | Standard TD",
            "snc_doc_fields": tos,
            "snc_base_fields": snc_base_fields,
            "topic_source": "standard",
            "topic_query_fields": None,
            "query_fields": ["TD"],
        })

        # Expanded Topics (TD + CKS)
        presets.append({
            "description": f"SNC+{tos_name} | {base_name} base | Expanded TD+CKS",
            "snc_doc_fields": tos,
            "snc_base_fields": snc_base_fields,
            "topic_source": "expanded",
            "topic_query_fields": ["TITLE", "DESCRIPTION", "combined_keyword_string"],
            "query_fields": ["TD"],
        })

        # Expanded Topics (TD + DHS)
        presets.append({
            "description": f"SNC+{tos_name} | {base_name} base | Expanded TD+DHS",
            "snc_doc_fields": tos,
            "snc_base_fields": snc_base_fields,
            "topic_source": "expanded",
            "topic_query_fields": ["TITLE", "DESCRIPTION", "dense_hyde_summary"],
            "query_fields": ["TD"],
        })

    return presets


# --- LP base (label_parent_expanded only) ---
SNC_DOC_EXPERIMENTS_LP = _build_presets(
    snc_base_fields=['label_parent_expanded'],
    base_name='LP'
)

# --- LP+RS base (label_parent_expanded + raw_scope) ---
SNC_DOC_EXPERIMENTS_LP_RS = _build_presets(
    snc_base_fields=['label_parent_expanded', 'raw_scope'],
    base_name='LP+RS'
)


# ============================================================================
#  RUNNER FUNCTIONS
# ============================================================================

def run_preset_group(experiments, models=None):
    """
    Runs a list of preset experiments with the specified models.

    Args:
        experiments (list): List of experiment config dicts.
        models (list): Models to use. Default: ['bm25'].
    """
    if models is None:
        models = ['bm25']

    for i, exp in enumerate(experiments, 1):
        print(f"\n{Style.BOLD}{Style.CYAN}[{i}/{len(experiments)}]{Style.RESET}")
        run_snc_doc_experiment(
            snc_doc_fields=exp["snc_doc_fields"],
            snc_base_fields=exp["snc_base_fields"],
            topic_source=exp["topic_source"],
            topic_query_fields=exp.get("topic_query_fields"),
            models=models,
            query_fields=exp.get("query_fields", ["TD"]),
            description=exp["description"],
        )


def run_all_presets(models=None):
    """
    Runs ALL preset experiment groups sequentially.

    Args:
        models (list): Models to use across all experiments. Default: ['bm25'].
    """
    if models is None:
        models = ['bm25']

    total = len(SNC_DOC_EXPERIMENTS_LP) + len(SNC_DOC_EXPERIMENTS_LP_RS)
    print(f"\n{Style.BOLD}{Style.GREEN}{'='*70}")
    print(f"  RUNNING ALL SNC DOCUMENT EXPANSION EXPERIMENTS")
    print(f"  Models: {models}")
    print(f"  Total experiments: {total}")
    print(f"{'='*70}{Style.RESET}\n")

    print(f"\n{Style.BOLD}--- LP BASE EXPERIMENTS (label_parent_expanded) ---{Style.RESET}")
    run_preset_group(SNC_DOC_EXPERIMENTS_LP, models=models)

    print(f"\n{Style.BOLD}--- LP+RS BASE EXPERIMENTS (label_parent_expanded + raw_scope) ---{Style.RESET}")
    run_preset_group(SNC_DOC_EXPERIMENTS_LP_RS, models=models)

    print(f"\n{Style.BOLD}{Style.GREEN}> ALL SNC DOC EXPANSION EXPERIMENTS COMPLETE!{Style.RESET}")


# ============================================================================
#  MAIN
# ============================================================================

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # CONFIGURE YOUR RUN HERE
    # -----------------------------------------------------------------------
    # Models to test (singles only, as specified).
    # Each model runs independently — no RRF between models.
    MODELS = [['bm25'], ['colbert'], ['embeddings']]

    # -----------------------------------------------------------------------
    # OPTION 1: Run ALL presets (7 TOS × 2 SNC base × 3 topic configs = 42
    #           per model, × 3 models = 126 total experiment configurations,
    #           each with 3 random seeds)
    # -----------------------------------------------------------------------
    for model in MODELS:
        print(f"\n{'#'*70}")
        print(f"# Running ALL experiments for model: {model}")
        print(f"{'#'*70}")
        run_all_presets(models=model)

    # -----------------------------------------------------------------------
    # OPTION 2: Run a specific group (uncomment one):
    # -----------------------------------------------------------------------
    # run_preset_group(SNC_DOC_EXPERIMENTS_LP, models=['bm25'])
    # run_preset_group(SNC_DOC_EXPERIMENTS_LP_RS, models=['colbert'])

    # -----------------------------------------------------------------------
    # OPTION 3: Run a single custom experiment (uncomment and modify):
    # -----------------------------------------------------------------------
    # run_snc_doc_experiment(
    #     snc_doc_fields=['title', 'summary'],
    #     snc_base_fields=['label_parent_expanded'],
    #     topic_source='expanded',
    #     topic_query_fields=['TITLE', 'DESCRIPTION', 'combined_keyword_string'],
    #     models=['bm25'],
    #     description="Custom: SNC+TS | LP base | Expanded TD+CKS | BM25",
    # )
