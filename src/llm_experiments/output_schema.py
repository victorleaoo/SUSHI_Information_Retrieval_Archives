OUTPUT_SCHEMA = """Output exactly two labeled sections and nothing else. No preamble, no markdown, no
commentary.

CORE_THEMES: <one paragraph, {{N_WORDS}} words>
RELATED_CONCEPTS: <comma-separated list of {{N_TERMS}} terms>"""


def render_output_schema(n_words: str, n_terms: str) -> str:
    return OUTPUT_SCHEMA.replace("{{N_WORDS}}", n_words).replace("{{N_TERMS}}", n_terms)
