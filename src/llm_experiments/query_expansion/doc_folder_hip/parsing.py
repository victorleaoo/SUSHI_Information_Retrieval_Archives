"""Parsers turning the raw text responses from prompts.py into structured fields."""

import re

_DOC_PATTERN = re.compile(r"DOC_(\d)\s*:\s*(.*?)(?=DOC_\d\s*:|\Z)", re.DOTALL)


def parse_documents_response(raw_response: str) -> dict:
    """DOC_1/DOC_2/DOC_3 blocks -> {"doc_1": "...", "doc_2": "...", "doc_3": "..."}"""
    parsed = {}
    for num, text in _DOC_PATTERN.findall(raw_response or ""):
        parsed[f"doc_{num}"] = text.strip()
    return parsed


def parse_folder_label_response(raw_response: str) -> dict:
    """SNC_CODES/LABEL_TEXT/SUBJECT_TERMS blocks -> lists of codes/lines/terms."""
    raw_response = raw_response or ""

    snc_codes = []
    codes_match = re.search(r"SNC_CODES:\s*(.*)", raw_response)
    if codes_match:
        snc_codes = [c.strip() for c in codes_match.group(1).split(",") if c.strip()]

    label_text = []
    label_match = re.search(r"LABEL_TEXT:\s*(.*?)(?=SUBJECT_TERMS:|\Z)", raw_response, re.DOTALL)
    if label_match:
        label_text = [line.strip() for line in label_match.group(1).strip().splitlines() if line.strip()]

    subject_terms = []
    terms_match = re.search(r"SUBJECT_TERMS:\s*(.*)", raw_response, re.DOTALL)
    if terms_match:
        terms_text = terms_match.group(1).strip().replace("\n", " ")
        subject_terms = [t.strip() for t in terms_text.split(",") if t.strip()]

    return {
        "snc_codes": snc_codes,
        "label_text": label_text,
        "subject_terms": subject_terms,
    }
