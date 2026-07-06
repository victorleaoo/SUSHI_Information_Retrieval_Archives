import random
from utils.llm_client import LLMClient
from .aug_base import extract_description_keywords, augment_no_docs

def find_neighbor_folders(folder_meta, all_folders_with_docs, folder_metadata_dict):
    """
    Finds neighbor folders with documents based on cascading hierarchy:
    1. Exact SNC
    2. 2-level SNC prefix (e.g., 'POL 15' -> 'POL')
    3. Parent SNC (e.g., 'POL')
    """
    target_snc = folder_meta.get('snc', '')
    if not target_snc or target_snc == 'Unknown':
        return [], "none"
        
    target_id = folder_meta.get('id', '')
    
    # Pre-compute target features
    snc_parts = target_snc.split()
    snc_2level = " ".join(snc_parts[:2]) if len(snc_parts) >= 2 else target_snc
    parent_snc = folder_meta.get('label_parent_snc', '')
    if not parent_snc: parent_snc = snc_parts[0] if snc_parts else ""

    exact_matches = []
    level2_matches = []
    parent_matches = []
    
    for n_id, doc_ids in all_folders_with_docs.items():
        if n_id == target_id: continue
        
        n_meta = folder_metadata_dict.get(n_id, {})
        n_snc = n_meta.get('snc', '')
        if not n_snc: continue
        
        # 1. Exact Match
        if n_snc == target_snc:
            exact_matches.append((n_id, doc_ids))
            continue
            
        # 2. 2-level Match
        n_snc_parts = n_snc.split()
        n_snc_2level = " ".join(n_snc_parts[:2]) if len(n_snc_parts) >= 2 else n_snc
        if n_snc_2level == snc_2level:
            level2_matches.append((n_id, doc_ids))
            continue
            
        # 3. Parent Match
        n_parent = n_meta.get('label_parent_snc', '')
        if not n_parent: n_parent = n_snc_parts[0] if n_snc_parts else ""
        if n_parent == parent_snc:
            parent_matches.append((n_id, doc_ids))

    # Cascade
    if len(exact_matches) > 0 and sum(len(docs) for _, docs in exact_matches) >= 3:
        return exact_matches, f"exact '{target_snc}'"
    if len(level2_matches) > 0 and sum(len(docs) for _, docs in level2_matches) >= 3:
        return level2_matches, f"2-level '{snc_2level}'"
    if len(parent_matches) > 0 and sum(len(docs) for _, docs in parent_matches) >= 3:
        return parent_matches, f"parent '{parent_snc}'"
        
    return [], "none"

def select_neighbor_docs(neighbor_list, max_docs=5, max_per_folder=2):
    """
    Selects up to `max_docs` documents from neighbors, capping at `max_per_folder` per folder.
    """
    random.seed(42)
    # neighbor_list is [(folder_id, [doc_id1, ...]), ...]
    selected_docs = []
    
    # Sort for deterministic behavior
    sorted_neighbors = sorted(neighbor_list, key=lambda x: x[0])
    
    # We want diversity, so we round-robin
    folder_queues = {n_id: sorted(docs) for n_id, docs in sorted_neighbors}
    for docs in folder_queues.values():
        random.shuffle(docs)
        
    counts = {n_id: 0 for n_id in folder_queues}
    
    while len(selected_docs) < max_docs:
        added_in_round = False
        for n_id, docs in folder_queues.items():
            if len(selected_docs) >= max_docs: break
            if counts[n_id] < max_per_folder and len(docs) > 0:
                doc = docs.pop(0)
                selected_docs.append((n_id, doc))
                counts[n_id] += 1
                added_in_round = True
        if not added_in_round:
            break
            
    return selected_docs

def augment_with_neighbors(folder_meta, selected_neighbor_docs, match_level, all_docs_metadata, folder_metadata_dict, llm_client: LLMClient):
    if len(selected_neighbor_docs) == 0:
        return augment_no_docs(folder_meta, llm_client)

    folder_label = folder_meta.get('title', '')
    snc = folder_meta.get('snc', 'Unknown')
    snc_code = "Unclassified" if snc == "Unknown" or not snc.strip() else snc
    expanded_snc = folder_meta.get('expanded_snc', folder_label) if snc_code == "Unclassified" else folder_meta.get('expanded_snc', '')
    
    start_date = folder_meta.get('start_date', 'Unknown')
    end_date = folder_meta.get('end_date', 'Unknown')
    date_range = f"{start_date} to {end_date}" if end_date != "Unknown" else f"{start_date} (end date not recorded)"

    neighbor_docs_str = ""
    for idx, (n_id, did) in enumerate(selected_neighbor_docs):
        doc_meta = all_docs_metadata.get(did, {})
        n_meta = folder_metadata_dict.get(n_id, {})
        n_label = n_meta.get('title', 'Unknown Folder')
        title = doc_meta.get('title', '')
        summary = doc_meta.get('summary', '')
        neighbor_docs_str += f"[Related Doc {idx+1}] From Folder: {n_label} | Title: {title} | Summary: {summary}\n"

    prompt = f"""You are an expert archivist and historian specializing in U.S.-Brazil diplomatic relations during the 1960s and 1970s. Write enriched metadata for an archival folder that has not been digitized.

TARGET FOLDER (not yet digitized):
- Folder Label: {folder_label}
- SNC Code: {snc_code}
- SNC Meaning: {expanded_snc}
- Date Range: {date_range}

CONTEXTUAL DOCUMENTS FROM RELATED FOLDERS:
The documents below come from OTHER folders with a related SNC classification (match level: {match_level}). They do NOT belong to the target folder. Use them to understand what topics, actors, and events are typically filed under this classification — but do not assume their specific events apply to the target folder.

{neighbor_docs_str}

Think through these steps before writing:
Step 1 — SNC PATTERNS: Based on the related documents, what recurring topics, actors, and events appear under this SNC code?
Step 2 — TARGET PERIOD SPECIFICITY: What events specifically occurred in {start_date}–{end_date} that align with this SNC topic? Use historical knowledge to situate the target folder in time.
Step 3 — LABEL DISTINCTION: The target folder is specifically labeled "{folder_label}". What distinguishes it from the related folders? Focus on the geographic or topical qualifier in the label.

Then output the following two sections:

DESCRIPTION: A paragraph of 100-150 words starting exactly with "This folder likely contains" combining insights from the related documents with the target folder's specific label and time period. Reference specific historical events and actors relevant to this exact label and period. Do not copy events directly from the related documents — use them as thematic context only.

KEYWORDS: A comma-separated list of 10-15 search terms derived from the target folder's SNC topic and the patterns observed in related documents.

Output only the DESCRIPTION and KEYWORDS sections. Do not output your step-by-step reasoning."""

    response = llm_client.generate(prompt)
    desc, kw = extract_description_keywords(response)
    return desc, kw, prompt, response
