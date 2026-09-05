"""Prompt template for describing what an undigitized folder's contents would concern,
based purely on its Subject-Numeric classification metadata.
"""

from src.llm_experiments.output_schema import render_output_schema

N_WORDS = "100-150"
N_TERMS = "15-20"

FOLDER_CONTEXT_PROMPT_TEMPLATE = """{{COLLECTION_CONTEXT}}

You are an expert archivist and historian of U.S.–Brazil relations. Below is the archival
classification of one physical folder. No documents from this folder are available. Your
task is to describe what records filed under this classification, in this collection and
this period, would concern — so that a researcher's query can be matched against this
folder even though its contents have not been digitized.

CLASSIFICATION:
- Folder Label: {{folder_label}}
- Subject-Numeric Code: {{snc}}
- Meaning: {{parent_expanded_snc}}
- Scope Note: {{scope_note}}
- Date Range: {{start_date}} to {{end_date}}

{{OUTPUT_SCHEMA}}

CORE_THEMES must begin with "This folder contains" and describe the thematic scope implied
by the classification, situated in Brazil in the years given by the date range. Draw on
established historical knowledge of that subject in that period: the institutions, actors,
policies and events that records of this kind would concern. Do NOT invent specific
documents, specific telegram numbers, or specific incidents you cannot ground in the
classification and the historical record.

RELATED_CONCEPTS must list specific historical entities, institutions, programs, treaties,
places, technologies and diplomatic terms that records under this classification, in this
period, would plausibly name. Include Portuguese terms where they are the form actually
used in the period."""

RETRY_SYSTEM_PROMPT = (
    "Your previous response for this task was incomplete: CORE_THEMES or RELATED_CONCEPTS "
    "came out null, empty, or too short. Provide complete, fully detailed answers for both "
    "fields — do not leave either one null, empty, or under 25 characters."
)


def render_folder_context_prompt(
    collection_context: str,
    folder_label: str,
    snc: str,
    parent_expanded_snc: str,
    scope_note: str,
    start_date: str,
    end_date: str,
) -> str:
    def display(value):
        return value if value is not None else "None"

    output_schema = render_output_schema(N_WORDS, N_TERMS)

    return (
        FOLDER_CONTEXT_PROMPT_TEMPLATE.replace("{{COLLECTION_CONTEXT}}", collection_context)
        .replace("{{folder_label}}", display(folder_label))
        .replace("{{snc}}", display(snc))
        .replace("{{parent_expanded_snc}}", display(parent_expanded_snc))
        .replace("{{scope_note}}", display(scope_note))
        .replace("{{start_date}}", display(start_date))
        .replace("{{end_date}}", display(end_date))
        .replace("{{OUTPUT_SCHEMA}}", output_schema)
    )
