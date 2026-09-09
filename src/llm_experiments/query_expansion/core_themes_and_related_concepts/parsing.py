"""Parses the raw CORE_THEMES/RELATED_CONCEPTS response and validates completeness."""

import re

MIN_CORE_THEMES_CHARS = 25

_CORE_THEMES_PATTERN = re.compile(r"CORE_THEMES:\s*(.*?)(?=RELATED_CONCEPTS:|\Z)", re.DOTALL)
_RELATED_CONCEPTS_PATTERN = re.compile(r"RELATED_CONCEPTS:\s*(.*)", re.DOTALL)


def parse_core_themes_response(raw_response: str) -> dict:
    raw_response = raw_response or ""

    core_themes = None
    match = _CORE_THEMES_PATTERN.search(raw_response)
    if match:
        text = match.group(1).strip()
        core_themes = text or None

    related_concepts = []
    match = _RELATED_CONCEPTS_PATTERN.search(raw_response)
    if match:
        terms_text = match.group(1).strip().replace("\n", " ")
        related_concepts = [t.strip() for t in terms_text.split(",") if t.strip()]

    return {"core_themes": core_themes, "related_concepts": related_concepts}


def is_valid_core_themes(result: dict) -> bool:
    parsed = result.get("parsed") or {}
    core_themes = parsed.get("core_themes")
    related_concepts = parsed.get("related_concepts")
    return (
        bool(core_themes)
        and len(core_themes) >= MIN_CORE_THEMES_CHARS
        and bool(related_concepts)
        and len(related_concepts) > 0
    )
