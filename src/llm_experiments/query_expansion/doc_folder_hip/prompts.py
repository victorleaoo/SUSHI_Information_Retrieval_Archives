"""Prompt templates for hypothetical-document and hypothetical-folder-label query
expansion, transcribed from query_llm_call.md. Placeholders `{{COLLECTION_CONTEXT}}`
and `{{query_text}}` are filled in via straight string replacement (see
`render_documents_prompt` / `render_folder_label_prompt`) to avoid clashing with
literal braces in the prompt text.
"""

DOCUMENTS_PROMPT_TEMPLATE = """{{COLLECTION_CONTEXT}}

You are an expert historian of U.S.–Brazil relations reconstructing the documentary record.
A researcher is looking for documents matching the information need below. Write three short
passages that read as if they were excerpts from actual documents in this collection that
would satisfy this need.

INFORMATION NEED:
{{query_text}}

Write each passage as the document itself, not as a description of it. Use the vocabulary,
naming conventions, abbreviations and phrasing of U.S. State Department reporting from Brazil
in the 1963-1973 period. Name the posts, officials, ministries, agencies, programs, parties
and places that such a document would name. Do not hedge, do not use conditional language,
and do not refer to the researcher or to the search. Start by "This document is about..."

Do not invent specific telegram numbers, file references or document identifiers.

Output exactly:

DOC_1: <60-90 words>
DOC_2: <60-90 words>
DOC_3: <60-90 words>

No other text."""

FOLDER_LABEL_PROMPT_TEMPLATE = """{{COLLECTION_CONTEXT}}

You are an expert archivist of the U.S. State Department Subject-Numeric filing system. A
researcher is looking for documents matching the information need below. Predict where in the
filing system such documents would have been filed.

INFORMATION NEED:
{{query_text}}

Records were filed by clerks at the time of creation according to the administrative subject
of the document, not according to its later historical interest. Choose codes on that basis:
which office handled this material, and under what routine subject heading would it have been
filed. Prefer the heading a clerk would have used over the one a historian would choose.

Output exactly:

SNC_CODES: <3 to 5 codes, most likely first, comma-separated, in the form "POL 15-1">
LABEL_TEXT: <one line per code, in the form "<code> <expanded meaning of the code> BRAZ <year or year range>">
SUBJECT_TERMS: <10-15 terms, comma-separated, in the register of folder labels and filing
headings: subject headings, office and agency names, country and place names. Not prose,
not full sentences.>

No other text."""


def _render(template: str, collection_context: str, query_text: str) -> str:
    return template.replace("{{COLLECTION_CONTEXT}}", collection_context).replace("{{query_text}}", query_text)


def render_documents_prompt(collection_context: str, query_text: str) -> str:
    return _render(DOCUMENTS_PROMPT_TEMPLATE, collection_context, query_text)


def render_folder_label_prompt(collection_context: str, query_text: str) -> str:
    return _render(FOLDER_LABEL_PROMPT_TEMPLATE, collection_context, query_text)
