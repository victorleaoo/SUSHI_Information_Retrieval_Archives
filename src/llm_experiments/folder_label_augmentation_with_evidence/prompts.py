"""Prompt template for describing a folder's contents using documentary evidence sampled
from a seed's ECF training set: the folder's own training documents when it has any,
otherwise documents borrowed from the nearest related folders (same SNC, then similar
SNC, then same box). Falls back to folder_label_augmentation's no-evidence template
(imported by generate.py) when a seed's sample yields no evidence at all for a folder.
"""

from src.llm_experiments.output_schema import render_output_schema

N_WORDS = "100-150"
N_TERMS = "15-20"

FOLDER_CONTEXT_WITH_EVIDENCE_PROMPT_TEMPLATE = """{{COLLECTION_CONTEXT}}

You are an expert archivist and historian of U.S.–Brazil relations. Below is the archival
classification of one physical folder, together with document evidence. Merge them into a
single folder-level description that will be indexed for retrieval.

CLASSIFICATION:
- Folder Label: {{folder_label}}
- Subject-Numeric Code: {{snc}}
- Meaning: {{parent_expanded_snc}}
- Scope Note: {{scope_note}}
- Date Range: {{start_date}} to {{end_date}}

{{EVIDENCE_BLOCK}}

{{OUTPUT_SCHEMA}}

CORE_THEMES must begin with "This folder contains" and combine the classification's scope
with the concrete evidence above, in State Department terminology of the era. Draw on
established historical knowledge to make explicit the geopolitical context, institutions
and policies implicit in this material, so that queries which do not use the folder's exact
wording can still match it. Do not invent specific incidents, telegram numbers or named
individuals that neither the evidence nor the historical record supports.

RELATED_CONCEPTS must list specific entities, institutions, programs, treaties, places and
diplomatic terms that are either present in the evidence or historically bound to this
subject and period. Include Portuguese terms where they are the form actually used."""


def render_own_evidence_block(doc_texts: list) -> str:
    lines = "\n".join(f"- {text}" for text in doc_texts)
    return f"DOCUMENTS IN THIS FOLDER:\n{lines}"


def render_related_evidence_block(relation_texts: list) -> str:
    """`relation_texts` is a list of (relation_name, doc_text) tuples, e.g.
    ("same_snc", "New CNTC Elections Scheduled — The document discusses...")."""
    lines = "\n".join(
        f"- [from folder with {relation} relation]: {text}" for relation, text in relation_texts
    )
    return (
        "DOCUMENTS FROM RELATED FOLDERS (CONTEXTUAL EVIDENCE ONLY):\n"
        "The following documents are NOT in this folder. They come from folders that are nearby in\n"
        "the archive or share this folder's classification, and are shown only to indicate the kind\n"
        "of material this part of the collection holds. Do not state or imply that these specific\n"
        "documents, incidents or communications are in this folder.\n"
        f"{lines}"
    )


def render_folder_context_with_evidence_prompt(
    collection_context: str,
    folder_label: str,
    snc: str,
    parent_expanded_snc: str,
    scope_note: str,
    start_date: str,
    end_date: str,
    evidence_block: str,
) -> str:
    def display(value):
        return value if value is not None else "None"

    output_schema = render_output_schema(N_WORDS, N_TERMS)

    return (
        FOLDER_CONTEXT_WITH_EVIDENCE_PROMPT_TEMPLATE.replace("{{COLLECTION_CONTEXT}}", collection_context)
        .replace("{{folder_label}}", display(folder_label))
        .replace("{{snc}}", display(snc))
        .replace("{{parent_expanded_snc}}", display(parent_expanded_snc))
        .replace("{{scope_note}}", display(scope_note))
        .replace("{{start_date}}", display(start_date))
        .replace("{{end_date}}", display(end_date))
        .replace("{{EVIDENCE_BLOCK}}", evidence_block)
        .replace("{{OUTPUT_SCHEMA}}", output_schema)
    )
