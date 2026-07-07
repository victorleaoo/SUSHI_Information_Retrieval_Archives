"""
SNC Specificity Classifier.

Based on data analysis (Phase 0), SNC codes fall into two groups:
- 'specific': The SNC label directly names thematic content (AGR, LAB, HLTH, etc.).
  Expansion strategy: bridge concepts and related terms anchored on the label.
- 'generic': The SNC is too broad to predict specific content (POL, E, SOC, ORG, PER, Unknown).
  Expansion strategy: rely on document-derived context; documents are essential.
"""

GENERIC_SNC_PARENTS = {'POL', 'E', 'SOC', 'ORG', 'PER', 'Unknown'}


def classify_snc_specificity(snc: str) -> str:
    """
    Classifies an SNC code as 'specific' or 'generic'.

    Args:
        snc: The SNC code string (e.g., 'AGR', 'POL 18', 'Unknown').

    Returns:
        'generic' if the SNC parent is in the broad/ambiguous set, else 'specific'.

    Examples:
        >>> classify_snc_specificity('AGR')
        'specific'
        >>> classify_snc_specificity('POL 18')
        'generic'
        >>> classify_snc_specificity('Unknown')
        'generic'
    """
    if snc == 'Unknown':
        return 'generic'
    parent = snc.split()[0]  # 'POL 18-2' → 'POL'
    if parent in GENERIC_SNC_PARENTS:
        return 'generic'
    return 'specific'


def classify_all_folders(folder_metadata: dict) -> dict:
    """
    Classifies every folder in the metadata dict.

    Args:
        folder_metadata: {folder_id: {snc: ..., ...}, ...}

    Returns:
        {folder_id: 'specific' | 'generic'}
    """
    return {
        folder_id: classify_snc_specificity(folder.get('snc', 'Unknown'))
        for folder_id, folder in folder_metadata.items()
    }
