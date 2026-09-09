"""Prompt template for restating a topic's information need in the language of the
records themselves (CORE_THEMES) and listing the specific entities such documents
would name (RELATED_CONCEPTS).
"""

from src.llm_experiments.output_schema import render_output_schema

N_WORDS = "80-120"
N_TERMS = "15-20"

CORE_THEMES_PROMPT_TEMPLATE = """{{COLLECTION_CONTEXT}}

You are an expert archivist and historian of U.S.–Brazil relations helping a researcher
search this collection. Below is the researcher's information need.

INFORMATION NEED:
{{query_text}}

{{OUTPUT_SCHEMA}}

CORE_THEMES must restate this information need as it would be expressed in the language of
the records themselves: State Department terminology of the 1963–1973 period, naming the
institutions, actors, policies and events that documents satisfying this need would discuss.
Write it as a description of the documents being sought, not as a question or a request.

RELATED_CONCEPTS must list specific historical entities, institutions, programs, treaties,
places, technologies and diplomatic terms that such documents would name. Include Portuguese
terms where they are the form actually used in the period. Do not include generic words and
do not include terms unrelated to this information need."""


def render_core_themes_prompt(collection_context: str, query_text: str) -> str:
    output_schema = render_output_schema(N_WORDS, N_TERMS)
    return (
        CORE_THEMES_PROMPT_TEMPLATE.replace("{{COLLECTION_CONTEXT}}", collection_context)
        .replace("{{query_text}}", query_text)
        .replace("{{OUTPUT_SCHEMA}}", output_schema)
    )
