"""
Pure text builders for the LLM augmentation: the extra query / folder-label text that the rankers receive.

No imports, so the web app (web_app/) can use them without pulling in run_generator's models (torch, ColBERT).
run_generator re-exports them, so its callers are unchanged.
"""

__all__ = ["build_augmentation_text", "build_core_themes_augmentation_text", "build_folder_label_augmentation_text"]


def build_augmentation_text(topic_entry, variant):
    """
    Builds the extra query text for a single topic from a doc_folder_hip entry.

    - 'DOC': the three hypothetical documents (doc_1/doc_2/doc_3), joined by ". ".
    - 'FL': the hypothetical folder label(s) — label_text lines plus subject_terms —
      joined by " " and ", " respectively. Raw SNC codes (e.g. "POL 15-1") are
      skipped: they are filing-system tokens, not query vocabulary, and BM25's
      alphanumeric-only cleaning would mangle them anyway.
    - 'AUG': 'DOC' text and 'FL' text concatenated with ". ".

    Returns "" if the requested pieces are missing (caller falls back to the
    plain original query in that case).
    """
    documents_parsed = topic_entry.get('documents', {}).get('parsed', {})
    folder_label_parsed = topic_entry.get('folder_label', {}).get('parsed', {})

    doc_text = ". ".join(
        documents_parsed[f'doc_{i}'] for i in (1, 2, 3) if documents_parsed.get(f'doc_{i}')
    )

    label_text = " ".join(folder_label_parsed.get('label_text', []))
    subject_terms = ", ".join(folder_label_parsed.get('subject_terms', []))
    fl_text = " ".join(t for t in (label_text, subject_terms) if t)

    if variant == 'DOC':
        return doc_text
    elif variant == 'FL':
        return fl_text
    elif variant == 'AUG':
        return ". ".join(t for t in (doc_text, fl_text) if t)
    return ""


def build_core_themes_augmentation_text(topic_entry):
    """
    Builds the extra query text for a single topic from a
    core_themes_and_related_concepts entry ('CT' variant): the core_themes
    paragraph plus related_concepts, joined the same way build_augmentation_text's
    'FL' variant joins label_text + subject_terms (space-joined prose, then a
    comma-joined term list appended).

    Returns "" if the parsed fields are missing (caller falls back to the
    plain original query in that case).
    """
    parsed = topic_entry.get('core_themes', {}).get('parsed', {})
    core_themes = parsed.get('core_themes') or ''
    related_concepts = ", ".join(parsed.get('related_concepts', []))
    return " ".join(t for t in (core_themes, related_concepts) if t)


def build_folder_label_augmentation_text(folder_entry):
    """
    Builds the extra folder-label text from a folder_label_augmentation entry:
    the folder's core_themes paragraph plus related_concepts, same composition
    as build_core_themes_augmentation_text.

    Returns "" if the parsed fields are missing (caller falls back to the
    plain original label in that case).
    """
    parsed = folder_entry.get('parsed', {})
    core_themes = parsed.get('core_themes') or ''
    related_concepts = ", ".join(parsed.get('related_concepts', []))
    return " ".join(t for t in (core_themes, related_concepts) if t)
