import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

"""
Experiment Runner for LLM-Expanded Folder Label Comparisons.

This script runs folder-level retrieval experiments comparing:
1. COMMON folder labels (from FoldersV1.3.json)
2. LLM-GENERATED folder labels (from expanded_snc.json, mapped to folders via SNC codes)

All experiments use all_folders_folder_label=True (single seed, folder-level retrieval).

FOLDER LABEL CONFIGURATIONS:
    Common Source ('common'):
        - 'label_parent_expanded': e.g., "POLITICAL AFFAIRS & RELATIONS: ELECTIONS"
        - 'raw_scope': Full scope text from the classification system
        - 'scope_truncated': Shortened scope text

    LLM Expanded Source ('llm_expanded'):
        - 'enhanced_folder_title': LLM-generated descriptive title
        - 'dense_embedding_summary': LLM-generated dense passage for embedding models
        - 'bm25_keywords': Concatenation of bureaucratic_jargon + historical_entities
          (key_figures and locations excluded — too repetitive across SNCs)

    Fallback: Folders with SNC='Unknown' or missing from expanded_snc.json use raw 'label'.

TOPIC QUERY CONFIGURATIONS:
    Standard ('standard'): Uses topics_output.txt with T/TD/TDN query modes.
    Expanded ('expanded'): Uses topics_expanded.txt with configurable fields:
        - 'TITLE', 'DESCRIPTION', 'NARRATIVE': Original topic fields
        - 'combined_keyword_string': Pre-concatenated BM25 keywords
        - 'dense_hyde_summary': LLM-generated dense passage for embedding queries

USAGE:
    # Run all preset experiments:
    python run_expanded_experiments.py

    # Or import and run specific configurations:
    from run_expanded_experiments import run_experiment
    run_experiment(folder_label_source='llm_expanded',
                   folder_label_fields=['enhanced_folder_title', 'dense_embedding_summary'],
                   topic_source='expanded',
                   topic_query_fields=['TITLE', 'DESCRIPTION', 'combined_keyword_string'],
                   models=['bm25', 'colbert'])
"""

from run_generator import RunGenerator, Style


# ============================================================================
#  CORE EXPERIMENT FUNCTION
# ============================================================================

def run_experiment(
    folder_label_source='common',
    folder_label_fields=None,
    topic_source='standard',
    topic_query_fields=None,
    models=None,
    rrf_input='docs',
    query_fields=None,
    description=""
):
    """
    Runs a single folder-level retrieval experiment.

    All experiments use all_folders_folder_label=True (single seed, folder-level).

    Args:
        folder_label_source (str): 'common' or 'llm_expanded'.
        folder_label_fields (list): Fields to concatenate for folder label.
        topic_source (str): 'standard' or 'expanded'.
        topic_query_fields (list): Fields for query construction (when topic_source='expanded').
        models (list): List of model names. Default: ['bm25'].
        rrf_input (str): 'docs' (early fusion) or 'folders' (late fusion).
        query_fields (list): Standard query field modes (e.g., ['TD']). Used when topic_source='standard'.
        description (str): Human-readable description printed before the run.
    """
    if models is None:
        models = ['bm25']
    if query_fields is None:
        query_fields = ['TD']

    print(f"\n{Style.BOLD}{Style.HEADER}{'='*70}{Style.RESET}")
    print(f"{Style.BOLD}{Style.GREEN}> EXPERIMENT: {description}{Style.RESET}")
    print(f"  Folder Source: {folder_label_source}")
    print(f"  Folder Fields: {folder_label_fields}")
    print(f"  Topic Source:  {topic_source}")
    print(f"  Topic Fields:  {topic_query_fields}")
    print(f"  Models:        {models}")
    print(f"  RRF Input:     {rrf_input}")
    print(f"{Style.BOLD}{Style.HEADER}{'='*70}{Style.RESET}\n")

    gen = RunGenerator(
        searching_fields=[['folderlabel']],
        query_fields=query_fields,
        run_type='all_documents',
        models=models,
        expansion=[],
        all_folders_folder_label=True,
        rrf_input=rrf_input,
        folder_label_source=folder_label_source,
        folder_label_fields=folder_label_fields,
        topic_source=topic_source,
        topic_query_fields=topic_query_fields,
    )

    gen.run_experiments()


# ============================================================================
#  PRESET EXPERIMENT CONFIGURATIONS
# ============================================================================

# --- COMMON FOLDER LABEL EXPERIMENTS ---

COMMON_EXPERIMENTS = [
    {
        "description": "Common: label_parent_expanded only | Standard Topics TD",
        "folder_label_source": "common",
        "folder_label_fields": ["label_parent_expanded"],
        "topic_source": "standard",
        "topic_query_fields": None,
        "query_fields": ["TD"],
    },
    {
        "description": "Common: label_parent_expanded + raw_scope | Standard Topics TD",
        "folder_label_source": "common",
        "folder_label_fields": ["label_parent_expanded", "raw_scope"],
        "topic_source": "standard",
        "topic_query_fields": None,
        "query_fields": ["TD"],
    },
    {
        "description": "Common: label_parent_expanded + scope_truncated | Standard Topics TD",
        "folder_label_source": "common",
        "folder_label_fields": ["label_parent_expanded", "scope_truncated"],
        "topic_source": "standard",
        "topic_query_fields": None,
        "query_fields": ["TD"],
    },
]

# --- LLM-EXPANDED FOLDER LABEL EXPERIMENTS ---

LLM_EXPERIMENTS = [
    {
        "description": "LLM: enhanced_folder_title only | Standard Topics TD",
        "folder_label_source": "llm_expanded",
        "folder_label_fields": ["label_parent_extended", "enhanced_folder_title"],
        "topic_source": "standard",
        "topic_query_fields": None,
        "query_fields": ["TD"],
    },
    {
        "description": "LLM: enhanced_folder_title + dense_embedding_summary | Standard Topics TD",
        "folder_label_source": "llm_expanded",
        "folder_label_fields": ["label_parent_extended", "enhanced_folder_title", "dense_embedding_summary"],
        "topic_source": "standard",
        "topic_query_fields": None,
        "query_fields": ["TD"],
    },
    {
        "description": "LLM: enhanced_folder_title + bm25_keywords | Standard Topics TD",
        "folder_label_source": "llm_expanded",
        "folder_label_fields": ["label_parent_extended", "enhanced_folder_title", "bm25_keywords"],
        "topic_source": "standard",
        "topic_query_fields": None,
        "query_fields": ["TD"],
    },
    {
        "description": "LLM: enhanced_folder_title + dense_embedding_summary + bm25_keywords | Standard Topics TD",
        "folder_label_source": "llm_expanded",
        "folder_label_fields": ["label_parent_extended","enhanced_folder_title", "dense_embedding_summary", "bm25_keywords"],
        "topic_source": "standard",
        "topic_query_fields": None,
        "query_fields": ["TD"],
    },
]

# --- EXPANDED TOPIC EXPERIMENTS (with LLM folder labels) ---

EXPANDED_TOPIC_EXPERIMENTS = [
    {
        "description": "LLM: enhanced_folder_title | Expanded Topics TD+CKS",
        "folder_label_source": "llm_expanded",
        "folder_label_fields": ["label_parent_extended", "enhanced_folder_title"],
        "topic_source": "expanded",
        "topic_query_fields": ["TITLE", "DESCRIPTION", "combined_keyword_string"],
        "query_fields": ["TD"],
    },
    {
        "description": "LLM: enhanced_folder_title + dense_embedding_summary | Expanded Topics TD+CKS",
        "folder_label_source": "llm_expanded",
        "folder_label_fields": ["label_parent_extended", "enhanced_folder_title", "dense_embedding_summary"],
        "topic_source": "expanded",
        "topic_query_fields": ["TITLE", "DESCRIPTION", "combined_keyword_string"],
        "query_fields": ["TD"],
    },
    {
        "description": "LLM: enhanced_folder_title + dense_embedding_summary | Expanded Topics TD+DHS",
        "folder_label_source": "llm_expanded",
        "folder_label_fields": ["label_parent_extended", "enhanced_folder_title", "dense_embedding_summary"],
        "topic_source": "expanded",
        "topic_query_fields": ["TITLE", "DESCRIPTION", "dense_hyde_summary"],
        "query_fields": ["TD"],
    },
    {
        "description": "LLM: enhanced_folder_title + bm25_keywords | Expanded Topics TD+CKS",
        "folder_label_source": "llm_expanded",
        "folder_label_fields": ["label_parent_extended", "enhanced_folder_title", "bm25_keywords"],
        "topic_source": "expanded",
        "topic_query_fields": ["TITLE", "DESCRIPTION", "combined_keyword_string"],
        "query_fields": ["TD"],
    },
    {
        "description": "LLM: enhanced_folder_title + dense_embedding_summary + bm25_keywords | Expanded Topics TD+DHS",
        "folder_label_source": "llm_expanded",
        "folder_label_fields": ["label_parent_extended", "enhanced_folder_title", "dense_embedding_summary", "bm25_keywords"],
        "topic_source": "expanded",
        "topic_query_fields": ["TITLE", "DESCRIPTION", "dense_hyde_summary"],
        "query_fields": ["TD"],
    },
    {
        "description": "Common: label_parent_expanded | Expanded Topics TD+CKS",
        "folder_label_source": "common",
        "folder_label_fields": ["label_parent_expanded", "scope_truncated"],
        "topic_source": "expanded",
        "topic_query_fields": ["TITLE", "DESCRIPTION", "combined_keyword_string"],
        "query_fields": ["TD"],
    },
    {
        "description": "Common: label_parent_expanded | Expanded Topics TD+DHS",
        "folder_label_source": "common",
        "folder_label_fields": ["label_parent_expanded", "scope_truncated"],
        "topic_source": "expanded",
        "topic_query_fields": ["TITLE", "DESCRIPTION", "dense_hyde_summary"],
        "query_fields": ["TD"],
    },
]


# ============================================================================
#  RUNNER FUNCTIONS
# ============================================================================

def run_preset_group(experiments, models=None, rrf_input='docs'):
    """
    Runs a list of preset experiments with the specified models.

    Args:
        experiments (list): List of experiment config dicts.
        models (list): Models to use. Default: ['bm25'].
        rrf_input (str): RRF fusion strategy.
    """
    if models is None:
        models = ['bm25']

    for i, exp in enumerate(experiments, 1):
        print(f"\n{Style.BOLD}{Style.CYAN}[{i}/{len(experiments)}]{Style.RESET}")
        run_experiment(
            folder_label_source=exp["folder_label_source"],
            folder_label_fields=exp["folder_label_fields"],
            topic_source=exp["topic_source"],
            topic_query_fields=exp.get("topic_query_fields"),
            models=models,
            rrf_input=rrf_input,
            query_fields=exp.get("query_fields", ["TD"]),
            description=exp["description"],
        )


def run_all_presets(models=None, rrf_input='docs'):
    """
    Runs ALL preset experiment groups sequentially.

    Args:
        models (list): Models to use across all experiments. Default: ['bm25'].
        rrf_input (str): RRF fusion strategy.
    """
    if models is None:
        models = ['bm25']

    print(f"\n{Style.BOLD}{Style.GREEN}{'='*70}")
    print(f"  RUNNING ALL PRESET EXPERIMENTS")
    print(f"  Models: {models} | RRF Input: {rrf_input}")
    print(f"{'='*70}{Style.RESET}\n")

    total = len(COMMON_EXPERIMENTS) + len(LLM_EXPERIMENTS) + len(EXPANDED_TOPIC_EXPERIMENTS)
    print(f"  Total experiments: {total}\n")

    print(f"\n{Style.BOLD}--- COMMON FOLDER LABEL EXPERIMENTS ---{Style.RESET}")
    run_preset_group(COMMON_EXPERIMENTS, models=models, rrf_input=rrf_input)

    print(f"\n{Style.BOLD}--- LLM-EXPANDED FOLDER LABEL EXPERIMENTS ---{Style.RESET}")
    run_preset_group(LLM_EXPERIMENTS, models=models, rrf_input=rrf_input)

    print(f"\n{Style.BOLD}--- EXPANDED TOPIC EXPERIMENTS ---{Style.RESET}")
    run_preset_group(EXPANDED_TOPIC_EXPERIMENTS, models=models, rrf_input=rrf_input)

    print(f"\n{Style.BOLD}{Style.GREEN}> ALL EXPERIMENTS COMPLETE!{Style.RESET}")


# ============================================================================
#  MAIN
# ============================================================================

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # CONFIGURE YOUR RUN HERE
    # -----------------------------------------------------------------------
    # Choose which models to test. Options: 'bm25', 'embeddings', 'colbert'
    # For multiple models, RRF fusion is applied automatically.
    MODELS = [['embeddings'], ['colbert'], ['bm25', 'embeddings'], ['bm25', 'embeddings', 'colbert']]

    # RRF fusion strategy: 'docs' (Early Fusion) or 'folders' (Late Fusion)
    RRF_INPUT = 'docs'

    # -----------------------------------------------------------------------
    # OPTION 1: Run ALL presets
    # -----------------------------------------------------------------------
    for model in MODELS:
        print("Running for models:")
        print(model)
        run_all_presets(models=model, rrf_input=RRF_INPUT)

    # -----------------------------------------------------------------------
    # OPTION 2: Run a specific group (uncomment one):
    # -----------------------------------------------------------------------
    # run_preset_group(COMMON_EXPERIMENTS, models=MODELS, rrf_input=RRF_INPUT)
    # run_preset_group(LLM_EXPERIMENTS, models=MODELS, rrf_input=RRF_INPUT)
    # run_preset_group(EXPANDED_TOPIC_EXPERIMENTS, models=MODELS, rrf_input=RRF_INPUT)

    # -----------------------------------------------------------------------
    # OPTION 3: Run a single custom experiment (uncomment and modify):
    # -----------------------------------------------------------------------
    # run_experiment(
    #     folder_label_source='llm_expanded',
    #     folder_label_fields=['enhanced_folder_title', 'dense_embedding_summary'],
    #     topic_source='expanded',
    #     topic_query_fields=['TITLE', 'DESCRIPTION', 'combined_keyword_string'],
    #     models=['bm25', 'colbert'],
    #     rrf_input='docs',
    #     description="Custom: LLM title+summary | Expanded TD+CKS | BM25+ColBERT",
    # )
